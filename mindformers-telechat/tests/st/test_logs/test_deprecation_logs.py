# Copyright 2025 Huawei Technologies Co., Ltd
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
Test deprecation logs of deprecated API.
How to run this:
pytest tests/st/test_logs/test_deprecation_logs.py
"""
import inspect
import pytest

from mindformers.utils import deprecated


@deprecated(reason="This method is rotten.", version="1.0.0")
def deprecated_method(arg1, arg2):
    """A fake deprecated method."""
    _ = arg1, arg2


@deprecated(reason="This class is rotten.", version="1.0.0")
class DeprecatedClass:
    """A fake deprecated class."""
    def __init__(self, arg1, arg2):
        _ = arg1, arg2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestDeprecationLogs:
    """A test class for testing deprecation logs."""

    def test_signature(self):
        """Test signature of fake APIs."""
        method_sign = inspect.signature(deprecated_method)
        assert str(method_sign) == "(arg1, arg2)"

        class_sign = inspect.signature(DeprecatedClass)
        assert str(class_sign) == "(arg1, arg2)"
