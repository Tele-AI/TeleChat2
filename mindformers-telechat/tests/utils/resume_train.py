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
"""resume training utils."""
import re
import os


def extract_loss_values(log_file_path):
    """extract loss values from log"""
    loss_values = []
    loss_pattern = re.compile(r'loss: (\d+\.\d+)')

    with open(log_file_path, 'r') as file:
        for line in file:
            match = loss_pattern.search(line)
            if match:
                loss_value = float(match.group(1))
                loss_values.append(loss_value)

    return loss_values


def get_file_mtime(file_path):
    return os.path.getmtime(file_path)
