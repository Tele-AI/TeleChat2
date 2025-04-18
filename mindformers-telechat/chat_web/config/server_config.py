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
"""Server Config"""
import yaml
from mindformers.tools.check_rules import check_yaml_depth_before_loading


class ServerConfig:
    """
    Server config

    Args:
        path (str):
            path of server config.
    """
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            check_yaml_depth_before_loading(f)
            f.seek(0)
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

    def __getitem__(self, item):
        return self.config[item]


default_config = ServerConfig('config/config.yaml').config
