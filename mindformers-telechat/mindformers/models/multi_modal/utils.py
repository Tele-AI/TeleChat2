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
utils for multi modal model
"""


class DataRecord:
    """Record data"""
    def __init__(self):
        self.data = {}

    def put(self, key, value, append=False):
        """put value into data, it allows appending to an existed value if assign append=True"""
        if key in self.data:
            if append:
                self.data[key].append(value)
            else:
                self.data[key] = value
        else:
            if append:
                self.data[key] = [value]
            else:
                self.data[key] = value

    def put_from_dict(self, dict_):
        for key, value in dict_.items():
            self.put(key, value)

    def get(self, key):
        return self.data.get(key)

    def has_key(self, key):
        return key in self.data

    def output(self, keys, format_="dict"):
        if format_ == "tuple":
            return tuple(self.get(key) for key in keys)

        if format_ == "dict":
            return {key: self.get(key) for key in keys}
        raise ValueError(f"format={format_} is not supported.")

    def clear(self):
        self.data = {}
