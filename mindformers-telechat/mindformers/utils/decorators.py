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
Common decorators of all the methods in MindFormers
"""
from functools import wraps
from deprecated import deprecated as origin_deprecated


def deprecated(reason: str = None, version: str = None):
    """
    A decorator for deprecated API that can keep the original signature of the API.

    Args:
        reason (str, optional): Reason message which documents the deprecation. Default: ``None``.
        version (str, optional): Version of the project which deprecates this API.
            If follow the Semantic Versioning, the version number has the format “MAJOR.MINOR.PATCH”.
            Default: ``None``.

    Returns:
        A decorator.
    """
    def decorator(obj):
        @wraps(obj)
        @origin_deprecated(reason=reason, version=version)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)
        return wrapper
    return decorator
