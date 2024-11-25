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
""" test cann binding cores."""

import pytest
from mindformers.utils.bit_array import int_to_binary_list, binary_list_to_int, BitArray

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_int_to_binary_list():
    """
    Feature: int_to_binary_list
    Description: int to binary list
    Expectation: No Exception
    """
    int_to_list_map = {
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 1, 1],
        4: [0, 1, 0, 0],
        5: [0, 1, 0, 1],
        6: [0, 1, 1, 0],
        7: [0, 1, 1, 1],
        8: [1, 0, 0, 0],
        9: [1, 0, 0, 1],
        10: [1, 0, 1, 0],
        11: [1, 0, 1, 1],
        12: [1, 1, 0, 0],
        13: [1, 1, 0, 1],
        14: [1, 1, 1, 0],
        15: [1, 1, 1, 1],
    }

    for v in range(16):
        bin_list = int_to_binary_list(v)
        assert bin_list == int_to_list_map[v]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_binary_list_to_int():
    """
    Feature: binary_list_to_int
    Description: binary list to int
    Expectation: No Exception
    """
    int_to_list_map = {
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 1, 1],
        4: [0, 1, 0, 0],
        5: [0, 1, 0, 1],
        6: [0, 1, 1, 0],
        7: [0, 1, 1, 1],
        8: [1, 0, 0, 0],
        9: [1, 0, 0, 1],
        10: [1, 0, 1, 0],
        11: [1, 0, 1, 1],
        12: [1, 1, 0, 0],
        13: [1, 1, 0, 1],
        14: [1, 1, 1, 0],
        15: [1, 1, 1, 1],
    }

    for v in range(16):
        bin_list = int_to_list_map[v]
        value = binary_list_to_int(bin_list)
        assert value == v


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bit_array_from_string():
    """
    Feature: bit array
    Description: test bit array from string
    Expectation: No Exception
    """
    bitarray = BitArray(32)
    bitarray.load_from_str("deadbeef")
    mark_list = bitarray.get_marked_index()
    expect_mark_list = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 25, 26, 27, 28, 30, 31]
    assert mark_list == expect_mark_list


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bit_array_setitem_and_getitem():
    """
    Feature: bit array
    Description: test bit array setitem and getitem
    Expectation: No Exception
    """
    bitarray = BitArray(32)
    assert bitarray[10] == 0
    bitarray[10] = 1
    assert bitarray[10] == 1
    assert bitarray[11] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bit_array_to_bytes_array():
    """
    Feature: bit array
    Description: test bit array from string
    Expectation: No Exception
    """
    bitarray = BitArray(32)
    bitarray[2] = 1
    bitarray[5] = 1
    bitarray[10] = 1
    bitarray[13] = 1
    bitarray[16] = 1
    bitarray[20] = 1
    bitarray[26] = 1
    bitarray[29] = 1
    bytes_array = bitarray.to_bytes_array()
    expect_bytes_array = [36, 17, 36, 36]
    assert bytes_array == expect_bytes_array
