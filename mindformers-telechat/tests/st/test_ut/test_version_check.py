# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test version check schedule."""
import pytest
from mindformers.tools.utils import is_version_le, is_version_ge


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_is_version_ge():
    """
    Feature: IsVersionGreaterEqual
    Description: Test IsVersionGreaterEqual
    Expectation: ValueError
    """
    assert is_version_ge("2.0.0rc1", '1.8.0')
    assert is_version_ge("1.8.1", '1.8.0')
    assert is_version_ge("1.8.0", '1.8.0')
    assert is_version_ge("2.0.0rc1", '2.0.0')
    assert not is_version_ge("1.8.0", '1.10.0')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_is_version_le():
    """
    Feature: IsVersionLessEqual
    Description: Test IsVersionLessEqual
    Expectation: ValueError
    """
    assert is_version_le("2.0.0rc1", '2.0.0')
    assert is_version_le("1.8.1", '2.0.0')
    assert is_version_le("1.8.0", '2.0.0rc1')
