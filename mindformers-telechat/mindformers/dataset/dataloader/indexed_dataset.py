# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Essentially re-written in entirety
#
# Added a dataloader class for resistering into MindFomers
# ============================================================================
"""Indexed Dataset builder and Indexed Dataset Loader"""

import shutil
import struct
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union
import numpy

from mindspore.dataset import GeneratorDataset

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float32 = 6
    float64 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        return key().itemsize

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        return numpy.int32


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype`
                constructed from reading bytes from the data file starting at `offset`.
        """


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """
    def __init__(self, bin_path: str) -> None:
        self.bin_path = bin_path

        self._bin_buffer_mmap = numpy.memmap(self.bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype`
                constructed from reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype`
                constructed from reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode='rb', buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence


class _IndexReader:
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """
    def __init__(self, idx_path: str, mutilmodal: bool) -> None:
        self.idx_path = idx_path
        self.bin_buffer_mmap = numpy.memmap(self.idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        logger.info(f"Load the {type(self).__name__} from {self.idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        logger.info(f"\tExtract the sequence lengths")
        time_begin = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        time_end = time.time()
        logger.info(f"\t> time elapsed: {time_end - time_begin:4f} seconds")

        logger.info(f"\tExtract the sequence pointers")
        time_begin = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        time_end = time.time()
        logger.info(f"\t> time elapsed: {time_end - time_begin:4f} seconds")

        logger.info(f"\tExtract the document indices")
        time_begin = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        time_end = time.time()
        logger.info(f"\t> time elapsed: {time_begin - time_end:4f} seconds")

        self.sequence_modes = None
        if mutilmodal:
            logger.info(f"\tExtract the sequence modes")
            time_begin = time.time()
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            time_end = time.time()
            logger.info(f"\t> time elapsed: {time_end - time_begin} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        logger.info(f"> total number of sequences: {len(self)}")
        logger.info(f"> total number of documents: {self.document_indices.shape[0] - 1}")

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class _IndexWriter:
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the index file
    """
    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
            self,
            sequence_lengths: List[int],
            sequence_modes: Optional[List[int]],
            document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            sequence_modes (Optional[List[int]]): The mode of each sequences

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = numpy.array(sequence_lengths, dtype=numpy.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = numpy.array(sequence_pointers, dtype=numpy.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = numpy.array(document_indices, dtype=numpy.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = numpy.array(sequence_modes, dtype=numpy.int8)
            self.idx_writer.write(sequence_modes.tobytes(order='C'))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class IndexedDataLoader:
    """Indexed DataLoader."""
    _default_column_names = ['input_ids']

    def __new__(cls,
                path_prefix: str,
                mutilmodal: bool = False,
                column_names: list = None,
                shuffle: bool = False,
                return_in_generator: bool = True,
                mmap: bool = True,
                num_shards: int = None,
                shard_id: int = None,
                **kwargs):
        """Return a GeneratorDataset or IndexedDataset.

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix.
            multimodal (bool): Whether the dataset is multimodal. Defaults to False.
            column_names (list): Column names contained in the created dataset.
            shuffle (bool): Whether to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.
            return_in_generator (bool): whether to return the GeneratorDataset
            mmap (bool): whether to mmap the .bin file.
            num_shards (int): device_num
            shard_id (int): rank_id

        Return:
            A GeneratorDataset object or IndexedDataset object.
        """
        column_names = cls._default_column_names if column_names is None else column_names
        kwargs = kwargs

        indexed_dataset = IndexedDataset(path_prefix=path_prefix, mutimodal=mutilmodal, mmap=mmap)

        if return_in_generator:
            gen_dataset = GeneratorDataset(indexed_dataset, column_names=column_names, shuffle=shuffle,
                                           num_shards=num_shards, shard_id=shard_id)
            return gen_dataset
        return indexed_dataset


class IndexedDataset:
    """Indexed Data Loader API returns the data as the source of the GeneratorDataset

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): whether to mmap the .bin file.
    """
    def __init__(self, path_prefix: str, mutimodal: bool = False, mmap: bool = True) -> None:
        self.path_prefix = path_prefix
        self.mutilmodal = mutimodal
        self.mmap = mmap
        self.index = None
        self.bin_reader = None

        self.initialize()

    def initialize(self) -> None:
        """Initialize the dataset"""
        idx_path = get_idx_path(self.path_prefix)
        bin_path = get_bin_path(self.path_prefix)

        if self.mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            self.bin_reader = _FileBinReader(bin_path)
        self.index = _IndexReader(idx_path, self.mutilmodal)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
            self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (Union[int, numpy.integer, slice]): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous

            TypeError: When the index is of an unexpected type

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index
                or index slice
        """
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        start, _, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("Slices into indexed_dataset must be contiguous")
        sequence_lengths = self.index.sequence_lengths[idx]
        sequence_modes = self.index.sequence_modes[idx] if self.mutilmodal else None
        sequence_offsets = list(accumulate(sequence_lengths))
        sequences = numpy.split(
            self.bin_reader.read(
                dtype=self.index.dtype,
                count=sum(sequence_lengths),
                offset=self.index.sequence_pointers[start],
            ),
            sequence_offsets[:-1],
        )
        return (sequences, sequence_modes) if sequence_modes is not None else sequences


class IndexedDataBuilder:
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """
    def __init__(self, bin_path: str, dtype: Type[numpy.number] = numpy.int32, multimodal: bool = False) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, np_array: numpy.ndarray, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            np_array (numpy.ndarray): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item()"""
        self.document_indices.append(len(self.sequence_lengths))

    def add_document(self, np_array: numpy.ndarray, lengths: List[int], modes: Optional[List[int]] = None) -> None:
        """Add an entire document to the dataset

        Args:
            np_array (numpy.ndarray): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]]): The modes for each item in the document. Default to None.
        """
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), self.multimodal)
        assert index.dtype == self.dtype

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        if self.multimodal:
            self.sequence_modes.extend(index.sequence_modes)

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): the prefix

    Returns:
        str : The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): the prefix

    Returns:
        str : The path to the data file
    """
    return path_prefix + ".bin"
