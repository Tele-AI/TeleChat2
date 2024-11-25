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

"""
BitArray to solve bit mask for core binding
"""


def int_to_binary_list(value: int, align_length: int = 4) -> list:
    """
    convert int value to binary list
    e.g. 13 => [1, 1, 0, 1]
    current only for 0 - 15

    Args:
        value (`int`):
            The int value to convert to binary list.
        align_length (`int`, *optional*, defaults to `4`):
            The align length for list, it will add 0 for small value

    Returns:
        The binary list with the value.
    """
    bin_list = []
    divider = value
    remainder = 0
    while True:
        remainder = divider % 2
        divider = int(divider / 2)
        bin_list.append(remainder)
        if divider == 0:
            break

    while len(bin_list) < align_length:
        bin_list.append(0)

    bin_list.reverse()
    return bin_list


def binary_list_to_int(bin_list: list) -> int:
    """
    convert binary list to int value
    e.g. [1, 1, 0, 1] => 13
    current only for 0 - 15

    Args:
        bin_list (`list`):
            The binary list represent to int value.

    Returns:
        The int value.
    """
    value = 0
    muliplier = 1
    bin_list.reverse()
    for v in bin_list:
        value = value + v * muliplier
        muliplier *= 2
    return value


def string_to_bit_list(array_string: str) -> list:
    """
    convert hex string to binary list
    e.g. "ff" => [1, 1, 1, 1, 1, 1, 1, 1]
        "deadbeef" => [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]

    Args:
        array_string (`str`):
            The binary list represent to int value.

    Returns:
        The binary list for the string.
    """
    bin_list = []
    for c in array_string:
        bit_list = int_to_binary_list(int(c, 16))
        bin_list += bit_list
    bin_list.reverse()
    return bin_list


class BitArray:
    """
    The bit array class to solve core mask string.

    Args:
        length(`int`, *optional*, defaults to `0`):
            The max bit length of the array.
    """

    def __init__(self, length: int = 0):
        self.bits = [0 for _ in range(length)]

    def load_from_str(self, array_string: str):
        """
        load bit array from hex string

        Args:
            array_string (`str`):
                The binary list represent to int value.

        Returns:
            NA.
        """
        self.bits = string_to_bit_list(array_string)

    def get_marked_index(self) -> list:
        """
        get the index list with value 1

        Args:
            NA.

        Returns:
            The index list.
        """
        marked_index_list = []
        for idx, item in enumerate(self.bits):
            if item == 1:
                marked_index_list.append(idx)
        return marked_index_list

    def to_bytes_array(self) -> list:
        """
        convert the bit array to byte array which is 8-bit elements

        Args:
            NA.

        Returns:
            The array values with bytes.
        """
        bytes_array = []
        slide_window_list = []
        self.bits.reverse()
        for idx, item in enumerate(self.bits):
            slide_window_list.append(item)
            if (idx + 1) % 8 == 0:
                value = binary_list_to_int(slide_window_list)
                slide_window_list.clear()
                bytes_array.append(value)
        self.bits.reverse()
        return bytes_array

    def __setitem__(self, index: int, value: int):
        """
        set the bit value with index

        Args:
            index (`int`):
                The index to set value.
            value (`int`):
                The value to set.

        Returns:
            NA.
        """
        self.bits[index] = value

    def __getitem__(self, index: int) -> int:
        """
        get the bit value with index

        Args:
            index (`int`):
                The index to get value.

        Returns:
            The value to get.
        """
        return self.bits[index]
