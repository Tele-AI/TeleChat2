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
"""test check_rules.py"""
import os
import unittest
import yaml
import pytest
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.check_rules import check_yaml_depth_before_loading, get_yaml_ast_depth


class TestCheckYamlDepthBeforeLoading(unittest.TestCase):
    """ A test class for testing check_yaml_depth_before_loading"""

    @classmethod
    def setUpClass(cls):
        cls.yaml_path = os.path.join(MindFormerBook.get_project_path(), 'configs', 'llama2', 'predict_llama2_7b.yaml')
        cls.yaml_str = """
        a:
            b:
                c:
                    d:
                        e:
                            f:
                                g:
                                    h:
                                        i:
                                            j:
                                                k:
                                                    l: "Too deep"
                            """

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_method(self):
        """test yaml load in different cases."""
        node = yaml.compose(self.yaml_str)
        assert get_yaml_ast_depth(node) == 12
        with pytest.raises(ValueError):
            assert check_yaml_depth_before_loading(self.yaml_str)
        with open(self.yaml_path, encoding='utf-8') as fp:
            node = yaml.compose(fp)
            assert get_yaml_ast_depth(node) == 3
            fp.seek(0)
            check_yaml_depth_before_loading(fp)
            fp.seek(0)
            cfg_dict = yaml.safe_load(fp)
            assert cfg_dict["model"]["model_config"]["num_layers"] == 32
            fp.seek(0)
            cfg_dict = yaml.load(fp, Loader=yaml.SafeLoader)
            assert cfg_dict["model"]["model_config"]["num_layers"] == 32
            fp.seek(0)
            cfg_dict = yaml.safe_load(fp.read())
            assert cfg_dict["model"]["model_config"]["num_layers"] == 32
