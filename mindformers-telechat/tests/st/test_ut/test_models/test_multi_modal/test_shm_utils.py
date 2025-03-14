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
"""test shared memory utils."""
import tempfile
import os
import unittest
from unittest.mock import patch, MagicMock
from multiprocessing import shared_memory
import numpy as np
from mindformers.models.multi_modal.shm_utils import create_shm, MAX_SHM_SIZE, release_shared_memory, \
    encode_shm_name_to_int64, SHM_NAME_PREFIX, SHM_NAME_MAX_LENGTH, encode_shape_to_int64, decode_shm_name_from_int64, \
    decode_shape_from_int64, get_data_from_shm


class TestCreateShm(unittest.TestCase):
    """Test create_shm."""
    def test_create_shm_success(self):
        """Test successful creation of shared memory."""
        temp_dir = tempfile.TemporaryDirectory()
        shm_name_save_path = os.path.join(temp_dir.name, "test_shm_names.txt")
        size = 1024
        shm = create_shm(size, shm_name_save_path)
        # Check if the shared memory is created
        self.assertIsInstance(shm, shared_memory.SharedMemory)
        self.assertTrue(os.path.exists(shm_name_save_path))
        # Check if the name is written to the file
        with open(shm_name_save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertIn(f"{shm.name}\n", lines)
        # Clean up shared memory and file
        shm.close()
        shm.unlink()

    def test_invalid_size_negative(self):
        """Test with invalid size (negative number)."""
        with self.assertRaises(ValueError) as cm:
            create_shm(-1, "dummy.txt")
        self.assertEqual(str(cm.exception), "Size must be a positive integer.")

    def test_invalid_size_zero(self):
        """Test with invalid size (zero)."""
        with self.assertRaises(ValueError) as cm:
            create_shm(0, "dummy.txt")
        self.assertEqual(str(cm.exception), "Size must be a positive integer.")

    def test_size_exceeds_limit(self):
        """Test when size exceeds the MAX_SHM_SIZE limit."""
        with self.assertRaises(ValueError) as cm:
            create_shm(MAX_SHM_SIZE + 1, "dummy.txt")
        self.assertEqual(
            str(cm.exception),
            f"Size exceeds the maximum allowed limit of {MAX_SHM_SIZE} bytes."
        )

    @patch("os.open")
    def test_file_write_error(self, mock_open_):
        """Test file write error handling."""
        mock_open_.side_effect = OSError("Mocked file write error")
        with self.assertRaises(RuntimeError) as cm:
            create_shm(1024, "dummy.txt")
        self.assertIn("Failed to create shared memory", str(cm.exception))

    @patch("multiprocessing.shared_memory.SharedMemory")
    def test_shared_memory_creation_error(self, mock_shm):
        """Test shared memory creation error handling."""
        mock_shm.side_effect = OSError("Mocked shared memory error")
        with self.assertRaises(RuntimeError) as cm:
            create_shm(1024, "dummy.txt")
        self.assertIn("Failed to create shared memory", str(cm.exception))


class TestReleaseSharedMemory(unittest.TestCase):
    """Test release_shared_memory."""
    @patch("multiprocessing.shared_memory.SharedMemory")
    def test_release_shared_memory_success(self, mock_shared_memory):
        """Test successful release of shared memory."""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"shm_name_1\nshm_name_2\n")
            file_path = temp_file.name

        # 模拟 SharedMemory 的行为
        mock_shm_instance = MagicMock()
        mock_shared_memory.return_value = mock_shm_instance

        # 调用函数
        release_shared_memory(file_path)

        # 验证 SharedMemory 的调用
        mock_shared_memory.assert_any_call(name="shm_name_1")
        mock_shared_memory.assert_any_call(name="shm_name_2")
        mock_shm_instance.close.assert_called()
        mock_shm_instance.unlink.assert_called()

        # 删除临时文件
        os.remove(file_path)

    def test_release_shared_memory_file_not_exist(self):
        """Test when the file does not exist."""
        non_existent_file = "non_existent_file.txt"
        with self.assertRaises(FileExistsError) as cm:
            release_shared_memory(non_existent_file)
        self.assertIn(f"{non_existent_file} does not exists.", str(cm.exception))

    @patch("multiprocessing.shared_memory.SharedMemory")
    def test_release_shared_memory_error_handling(self, mock_shared_memory):
        """Test error handling during shared memory release."""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"shm_name_1\n")
            file_path = temp_file.name

        # 模拟 SharedMemory 的异常行为
        mock_shared_memory.side_effect = FileNotFoundError("Mocked shared memory not found.")

        with self.assertRaises(RuntimeError) as cm:
            release_shared_memory(file_path)

        self.assertIn("release share memory error", str(cm.exception))
        self.assertIn("Mocked shared memory not found.", str(cm.exception))

        # 删除临时文件
        os.remove(file_path)


class TestEncodeShmNameToInt64(unittest.TestCase):
    """Test encode_shm_name_to_int64."""
    def test_valid_name(self):
        """Test encoding with a valid shared memory name."""
        name = SHM_NAME_PREFIX + "abcdef"
        result = encode_shm_name_to_int64(name)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)  # 确保结果是负数

    def test_max_length_name(self):
        """Test encoding with a maximum length shared memory name."""
        name = SHM_NAME_PREFIX + "a" * (SHM_NAME_MAX_LENGTH - len(SHM_NAME_PREFIX))
        result = encode_shm_name_to_int64(name)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)

    def test_name_exceeds_max_length(self):
        """Test encoding with a name exceeding the maximum length."""
        name = SHM_NAME_PREFIX + "a" * (SHM_NAME_MAX_LENGTH - len(SHM_NAME_PREFIX) + 1)
        with self.assertRaises(ValueError) as cm:
            encode_shm_name_to_int64(name)
        self.assertIn("The shared_memory name is too long", str(cm.exception))

    def test_invalid_character(self):
        """Test encoding with a name containing invalid characters."""
        name = SHM_NAME_PREFIX + "invalid$"
        with self.assertRaises(ValueError) as cm:
            encode_shm_name_to_int64(name)
        self.assertIn("Invalid character", str(cm.exception))

    def test_empty_name(self):
        """Test encoding with an empty shared memory name."""
        name = SHM_NAME_PREFIX
        result = encode_shm_name_to_int64(name)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)


class TestEncodeShapeToInt64(unittest.TestCase):
    """Test encode_shape_to_int64."""
    def test_valid_shape_2d(self):
        """Test encoding a valid 2D shape."""
        shape = (512, 256)
        result = encode_shape_to_int64(shape)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)

    def test_valid_shape_3d(self):
        """Test encoding a valid 3D shape."""
        shape = (128, 64, 32)
        result = encode_shape_to_int64(shape)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)  # 确保结果是负数

    def test_valid_shape_4d(self):
        """Test encoding a valid 4D shape."""
        shape = (64, 3, 224, 224)
        result = encode_shape_to_int64(shape)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)

    def test_valid_shape_5d(self):
        """Test encoding a valid 5D shape."""
        shape = (32, 64, 3, 128, 128)
        result = encode_shape_to_int64(shape)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)

    def test_invalid_shape_dimensions(self):
        """Test encoding with invalid number of dimensions."""
        shape = (128,)  # Only 1 dimension
        with self.assertRaises(ValueError) as cm:
            encode_shape_to_int64(shape)
        self.assertIn("Shape must have 3 to 5 dimensions", str(cm.exception))

    def test_dimension_too_large(self):
        """Test encoding with a dimension size that exceeds the limit."""
        shape = (1 << 18, 64, 32)  # First dimension exceeds the maximum allowed
        with self.assertRaises(ValueError) as cm:
            encode_shape_to_int64(shape)
        self.assertIn("Dimension", str(cm.exception))
        self.assertIn("is too large", str(cm.exception))

    def test_edge_case_large_dimensions(self):
        """Test encoding with large dimensions near the limit."""
        shape = (1 << 13 - 1, 1 << 14 - 1, 3, 1 << 14 - 1, 1 << 14 - 1)
        result = encode_shape_to_int64(shape)
        self.assertIsInstance(result, int)
        self.assertLess(result, 0)


class TestDecodeShmNameFromInt64(unittest.TestCase):
    """Test encode_shm_name_to_int64."""
    def test_valid_decoding(self):
        """Test decoding a valid encoded shared memory name."""
        shm_name = SHM_NAME_PREFIX + "abcdef"
        encoded_value = encode_shm_name_to_int64(shm_name)
        decoded_name = decode_shm_name_from_int64(encoded_value)
        self.assertEqual(decoded_name, shm_name)

    def test_invalid_checksum(self):
        """Test decoding with an invalid checksum."""
        shm_name = SHM_NAME_PREFIX + "abcdef"
        encoded_value = encode_shm_name_to_int64(shm_name)
        tampered_value = encoded_value - 1  # Alter the encoded value to break checksum
        with self.assertRaises(ValueError) as cm:
            decode_shm_name_from_int64(tampered_value)
        self.assertIn("Checksum verification failed", str(cm.exception))

    def test_edge_case_min_length(self):
        """Test decoding the shortest possible shared memory name."""
        shm_name = SHM_NAME_PREFIX
        encoded_value = encode_shm_name_to_int64(shm_name)
        decoded_name = decode_shm_name_from_int64(encoded_value)
        self.assertEqual(decoded_name, shm_name)

    def test_edge_case_max_length(self):
        """Test decoding the longest possible shared memory name."""
        shm_name = SHM_NAME_PREFIX + "a" * (SHM_NAME_MAX_LENGTH - len(SHM_NAME_PREFIX))
        encoded_value = encode_shm_name_to_int64(shm_name)
        decoded_name = decode_shm_name_from_int64(encoded_value)
        self.assertEqual(decoded_name, shm_name)

    def test_invalid_input_type(self):
        """Test decoding with an invalid input type."""
        invalid_input = "not_an_int"
        with self.assertRaises(TypeError):
            decode_shm_name_from_int64(invalid_input)

    def test_invalid_character_index(self):
        """Test decoding with an invalid character index."""
        shm_name = SHM_NAME_PREFIX + "a"
        encoded_value = encode_shm_name_to_int64(shm_name)
        with self.assertRaises(ValueError) as cm:
            decode_shm_name_from_int64(-encoded_value)
        self.assertIn("Decode shared_memory name error, invalid character index", str(cm.exception))


class TestDecodeShapeFromInt64(unittest.TestCase):
    """Test decode_shape_from_int64."""
    def test_valid_decoding(self):
        """Test decoding a valid encoded shape."""
        shape = (64, 128, 256)
        encoded_value = encode_shape_to_int64(shape)
        decoded_shape = decode_shape_from_int64(encoded_value)
        self.assertEqual(decoded_shape, list(shape))

    def test_invalid_checksum(self):
        """Test decoding with an invalid checksum."""
        shape = (32, 64, 128)
        encoded_value = encode_shape_to_int64(shape)
        tampered_value = encoded_value - 1  # Alter the encoded value to break checksum
        with self.assertRaises(ValueError) as cm:
            decode_shape_from_int64(tampered_value)
        self.assertIn("Checksum does not match", str(cm.exception))


class TestGetDataFromShm(unittest.TestCase):
    """Test get_data_from_shm."""
    def test_valid(self):
        """Test get data from shared memory."""
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        shm_name_save_path = os.path.join(path, "test_shm_names.txt")
        size = (64, 128, 256)
        array = np.ones(size)
        # create shared memory
        shm = create_shm(array.nbytes, shm_name_save_path)
        shared_array = np.ndarray(array.shape, dtype=np.float32, buffer=shm.buf)
        shared_array[:] = array
        # encode name and shape to value
        shm_name = encode_shm_name_to_int64(shm.name)
        shape_value = encode_shape_to_int64(array.shape)
        decode_array = get_data_from_shm(shm_name, shape_value, dtype=np.float32)
        assert np.sum(np.abs(decode_array - array)) == 0
