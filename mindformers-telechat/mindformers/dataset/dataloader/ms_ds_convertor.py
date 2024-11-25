# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ms ds converter"""

import weakref
from typing import (
    List,
    Optional,
    Union, Iterable,
)
import mindspore as ms
import numpy as np


def to_ms_dataset(
        self,
        batch_size: Optional[int] = None,
        columns: Optional[Union[str, List[str]]] = None,
        shuffle: bool = None,
        label_cols: Optional[Union[str, List[str]]] = None,
        num_workers: int = 1,
        schema: Optional[Union["Schema", str]] = None,
        num_samples: Optional[int] = None,
        sampler: Optional[Union["Sampler", Iterable]] = None,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        python_multiprocessing: Optional[bool] = True,
        max_rowsize: Optional[int] = 6,
):
    """Create a `ms.dataset.GeneratorDataset` from the underlying Dataset. This `ms.dataset.GeneratorDataset` will load
    and collate batches from the Dataset, and is suitable for passing to methods like `model.fit()`.
     The dataset will yield `dicts` for both inputs and labels unless the `dict` would contain only a single key, in
     which case a raw `ms.Tensor` is yielded instead.

    Args:
        batch_size (`int`, *optional*):
            Size of batches to load from the dataset. Defaults to `None`, which implies that the dataset won't be
            batched, but the returned dataset can be batched later with `ms_dataset.batch(batch_size)`.
        columns (`List[str]` or `str`, *optional*):
            Dataset column(s) to load in the `ms.dataset.GeneratorDataset`.
            Column names that are created by the `collate_fn` and that do not exist in the original dataset can be used.
        shuffle(`bool`, defaults to `None`):
            Shuffle the dataset order when loading. Recommended `True` for training, `False` for
            validation/evaluation.
        label_cols (`List[str]` or `str`, defaults to `None`):
            Dataset column(s) to load as labels.
            Note that many models compute loss internally rather than letting Keras do it, in which case
            passing the labels here is optional, as long as they're in the input `columns`.
        num_workers (`int`, defaults to `1`):
            Number of workers to use for loading the dataset. Only supported on Python versions >= 3.8.
        schema (Union[str, Schema], optional):
            Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`mindspore.dataset.Schema` are
            acceptable. Default: ``None`` .
        num_samples (int, optional):
            The number of samples to be included in the dataset.
            Default: ``None`` , all images.
        sampler (Union[Sampler, Iterable], optional):
            Object used to choose samples from the dataset. Random accessible
            input is required. Default: ``None`` , expected order behavior shown in the table below.
        num_shards (int, optional):
            Number of shards that the dataset will be divided into. Default: ``None`` .
            Random accessible input is required. When this argument is specified, `num_samples` reflects the maximum
            sample number of per shard.
        shard_id (int, optional):
            The shard ID within `num_shards` . Default: ``None`` .
            This argument must be specified only when `num_shards` is also specified.
            Random accessible input is required.
        python_multiprocessing (bool, optional):
            Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: ``True``.
        max_rowsize(int, optional):
            Maximum size of row in MB that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. This is only
            used if python_multiprocessing is set to True. Default: 16.

    Returns:
        `ms.dataset.GeneratorDataset`

    Examples:
        >>> ds_train = ds["train"].to_ms_dataset(
        ...    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
        ...    shuffle=True,
        ...    batch_size=16,
        ... )
    """
    if label_cols and not columns:
        raise ValueError("Cannot specify label_cols without specifying columns!")
    if label_cols is None:
        label_cols = []
    elif isinstance(label_cols, str):
        label_cols = [label_cols]
    if len(set(label_cols)) < len(label_cols):
        raise ValueError("List of label_cols contains duplicates.")
    if columns:
        if isinstance(columns, str):
            columns = [columns]
        if len(set(columns)) < len(columns):
            raise ValueError("List of columns contains duplicates.")
    else:
        columns = []
    if self.format["type"] not in ["custom", "numpy"]:
        dataset = self.with_format("numpy")
    else:
        dataset = self

    output_signature, _ = get_ms_output_signature(dataset)

    if "labels" in output_signature:
        if ("label_ids" in columns or "label" in columns) and "labels" not in columns:
            columns = [col for col in columns if col not in ["label_ids", "label"]] + ["labels"]
        if ("label_ids" in label_cols or "label" in label_cols) and "labels" not in label_cols:
            label_cols = [col for col in label_cols if col not in ["label_ids", "label"]] + ["labels"]

    for col in columns:
        if col not in output_signature:
            raise ValueError(f"Column {col} not found in dataset!")

    for col in label_cols:
        if col not in output_signature:
            raise ValueError(f"Label column {col} not found in dataset!")

    column_names = columns + label_cols
    if num_workers >= 1:
        ms_dataset = ms.dataset.GeneratorDataset(
            MSDatasetAdaptor(dataset, column_names),
            column_names=column_names,
            schema=schema,
            num_samples=num_samples,
            num_parallel_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
            num_shards=num_shards,
            shard_id=shard_id,
            python_multiprocessing=python_multiprocessing,
            max_rowsize=max_rowsize,
        )
    else:
        raise ValueError("num_workers must be >= 1")

    if batch_size:
        ms_dataset = ms_dataset.batch(batch_size)

    # ms use default prefetch strategy

    # Remove a reference to the open Arrow file on delete
    def cleanup_callback(ref):
        dataset.__del__()
        # pylint: disable=W0212
        self._TF_DATASET_REFS.remove(ref)
    # pylint: disable=W0212
    self._TF_DATASET_REFS.add(weakref.ref(ms_dataset, cleanup_callback))

    return ms_dataset


class MSDatasetAdaptor:
    """
        MS dataset adaptor
        Args:
            dataset (Datasets):
                huggingface dataset
            col_names (list):
                list of col names
        Returns:
            A dataset for MSDatasetAdaptor.
    """
    def __init__(self, dataset, col_names=None):
        self.dataset = dataset
        if col_names:
            self.col_names = col_names
        else:
            self.col_names = dataset.column_names
        for col_name in self.col_names:
            setattr(self, col_name, dataset[col_name])

    def __getitem__(self, idx):
        return [getattr(self, col_name)[idx] for col_name in self.col_names]

    def __len__(self):
        return len(self.dataset)


def get_ms_output_signature(dataset):
    """get ms output signature"""
    # pylint: disable=C1801
    if len(dataset) == 0:
        raise ValueError("Unable to get the output signature because the dataset is empty.")

    ms_columns_to_signatures = {}
    np_columns_to_dtypes = {}
    first_dataset_index = 0
    for column in dataset[first_dataset_index].keys():
        raw_arrays = [dataset[first_dataset_index][column]]
        # In case the collate_fn returns something strange
        np_arrays = []
        for array in raw_arrays:
            if isinstance(array, np.ndarray):
                np_arrays.append(array)
            elif isinstance(array, ms.Tensor):
                np_arrays.append(array.numpy())
            else:
                np_arrays.append(np.array(array))
        if np.issubdtype(np_arrays[0].dtype, np.integer) or np_arrays[0].dtype == bool:
            ms_dtype = ms.int64
            np_dtype = np.int64
        elif np.issubdtype(np_arrays[0].dtype, np.number):
            ms_dtype = ms.float32
            np_dtype = np.float32
        elif np_arrays[0].dtype.kind == "U":  # Unicode strings
            np_dtype = np.unicode_
            ms_dtype = ms.string
        else:
            raise RuntimeError(
                f"Unrecognized array dtype {np_arrays[0].dtype}. \n"
                "Nested types and image/audio types are not supported yet."
            )
        ms_columns_to_signatures[column] = ms_dtype
        np_columns_to_dtypes[column] = np_dtype

    return ms_columns_to_signatures, np_columns_to_dtypes
