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
"""path utils"""

import os
from pathlib import Path


def get_parent_directory(cur_path: Path, levels_up: int) -> Path:
    """get parent directory with levels up"""
    if isinstance(cur_path, str):
        cur_path = Path(cur_path)

    for _ in range(levels_up):
        cur_path = cur_path.parent
    return cur_path


def get_target_file_path(cur_path: Path, levels_up: int, target_path: Path):
    """get target file path with levels up and target path"""
    cur_path = get_parent_directory(cur_path, levels_up)
    return os.path.join(cur_path, target_path)
