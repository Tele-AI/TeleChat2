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
"""error"""


class SAPPError(ValueError):
    """SAPP Error"""


def _raise_sapp_error_general(err, flag):
    raise SAPPError("General SAPP Error")


def _assert_sapp(test: bool, msg: str):
    if not test:
        raise SAPPError(msg)


def _check_in_bounds(n, n_desc: str, lower_bound, higher_bound):
    _assert_sapp(n >= lower_bound,
                 n_desc + " " + str(n) + " should be higher than " + str(lower_bound))
    _assert_sapp(n <= higher_bound,
                 n_desc + " " + str(n) + " should be lower than " + str(higher_bound))
