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
"""test qkv concat check in safetensors format"""
import json
import os
import shutil
import tempfile
import pytest

import numpy as np
from safetensors.numpy import save_file

from mindformers import LlamaForCausalLM, ChatGLM2Model
from mindformers.tools.register import MindFormerConfig
from mindformers.utils import validate_qkv_concat


class TestValidateQKVConcat:
    """A test class for testing qkv weights check"""
    def setup_class(self):
        """construct fake safetensors data"""
        temp_work_dir = tempfile.mkdtemp()
        weight_path_noconcat = os.path.join(temp_work_dir, "weight_path_noconcat")
        os.makedirs(weight_path_noconcat)
        weight_path_concat = os.path.join(temp_work_dir, "weight_path_concat")
        os.makedirs(weight_path_concat)

        # 1. construct noconcat fake data
        fake_dict_q = {"model.layers.0.attention.wq.weight": np.array([0])}
        fake_dict_k = {"model.layers.0.attention.wk.weight": np.array([1])}
        fake_dict_v = {"model.layers.0.attention.wv.weight": np.array([2])}
        fake_dict_w1 = {"model.layers.0.feed_forward.w1.weight": np.array([3])}
        fake_dict_w3 = {"model.layers.0.feed_forward.w3.weight": np.array([4])}
        fake_noconcat_map = {"model.layers.0.attention.wq.weight": "model-00001-of-00005.safetensors",
                             "model.layers.0.attention.wk.weight": "model-00002-of-00005.safetensors",
                             "model.layers.0.attention.wv.weight": "model-00003-of-00005.safetensors",
                             "model.layers.0.feed_forward.w1.weight": "model-00004-of-00005.safetensors",
                             "model.layers.0.feed_forward.w3.weight": "model-00005-of-00005.safetensors"}
        fake_file_1 = os.path.join(weight_path_noconcat, "model-00001-of-00005.safetensors")
        fake_file_2 = os.path.join(weight_path_noconcat, "model-00002-of-00005.safetensors")
        fake_file_3 = os.path.join(weight_path_noconcat, "model-00003-of-00005.safetensors")
        fake_file_4 = os.path.join(weight_path_noconcat, "model-00004-of-00005.safetensors")
        fake_file_5 = os.path.join(weight_path_noconcat, "model-00005-of-00005.safetensors")
        fake_json_file_noconcat = os.path.join(weight_path_noconcat, 'param_name_map.json')

        save_file(tensor_dict=fake_dict_q, filename=fake_file_1)
        os.chmod(fake_file_1, 0o750)
        save_file(tensor_dict=fake_dict_k, filename=fake_file_2)
        os.chmod(fake_file_2, 0o750)
        save_file(tensor_dict=fake_dict_v, filename=fake_file_3)
        os.chmod(fake_file_3, 0o750)
        save_file(tensor_dict=fake_dict_w1, filename=fake_file_4)
        os.chmod(fake_file_4, 0o750)
        save_file(tensor_dict=fake_dict_w3, filename=fake_file_5)
        os.chmod(fake_file_5, 0o750)

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(fake_json_file_noconcat, flags_, 0o750), 'w') as f:
            json.dump(fake_noconcat_map, f)

        # 2. construct concat fake data
        fake_dict_qkv = {"model.layers.0.attention.w_qkv.weight": np.array([0, 1, 2])}
        fake_dict_gate_hidden = {"model.layers.0.attention.w_gate_hidden.weight": np.array([3, 4])}
        fake_concat_map = {"model.layers.0.attention.w_qkv.weight": "model-00001-of-00002.safetensors",
                           "model.layers.0.attention.w_gate_hidden.weight": "model-00002-of-00002.safetensors"}
        fake_file_qkv = os.path.join(weight_path_concat, "model-00001-of-00002.safetensors")
        fake_file_gate_hidden = os.path.join(weight_path_concat, "model-00002-of-00002.safetensors")
        fake_json_file_concat = os.path.join(weight_path_concat, 'param_name_map.json')

        save_file(tensor_dict=fake_dict_qkv, filename=fake_file_qkv)
        os.chmod(fake_file_qkv, 0o750)
        save_file(tensor_dict=fake_dict_gate_hidden, filename=fake_file_gate_hidden)
        os.chmod(fake_file_gate_hidden, 0o750)

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(fake_json_file_concat, flags_, 0o750), 'w') as f:
            json.dump(fake_concat_map, f)

        # 3. create mindformers config object
        model_config_noconcat = {'qkv_concat': False}
        self.config_noconcat = MindFormerConfig(**model_config_noconcat)
        model_config_concat = {'qkv_concat': True}
        self.config_concat = MindFormerConfig(**model_config_concat)

        self.temp_work_dir = temp_work_dir
        self.weight_path_noconcat = weight_path_noconcat
        self.weight_path_concat = weight_path_concat
        self.log_file_path = "output/log/rank_0/info.log"
        self.model = LlamaForCausalLM


    def teardown_class(self):
        """remove fake safetensors directory"""
        shutil.rmtree(self.temp_work_dir)

    def test_qkv_config_true_with_concat_weights(self):
        """test validate qkv concat config which is true and weights with concatenate"""
        qkv_concat_config = self.config_concat.qkv_concat
        validate_qkv_concat(self.model, qkv_concat_config, self.weight_path_concat)

        log_content = self._get_log_content()
        self._reset_log_content()

        assert "The qkv concat check succeed! The qkv in the model weights has been concatenated" in log_content

    def test_qkv_config_false_with_noconcat_weights(self):
        """test validate qkv concat config which is false and weights with no concatenate"""
        qkv_concat_config = self.config_noconcat.qkv_concat
        validate_qkv_concat(self.model, qkv_concat_config, self.weight_path_noconcat)

        log_content = self._get_log_content()
        self._reset_log_content()

        assert "The qkv concat check succeed! The qkv in the model weights has been not concatenated" in log_content

    def test_qkv_config_true_with_noconcat_weights(self):
        """test validate qkv concat config which is true and weights with no concatenate"""
        qkv_concat_config = self.config_concat.qkv_concat
        with pytest.raises(ValueError, match=r"The qkv concat check failed! "
                                             r"The qkv in the model weights has been not concatenated"):
            validate_qkv_concat(self.model, qkv_concat_config, self.weight_path_noconcat)

    def test_qkv_config_false_with_concat_weights(self):
        """test validate qkv concat config which is false and weights with concatenate"""
        qkv_concat_config = self.config_noconcat.qkv_concat
        with pytest.raises(ValueError, match=r"The qkv concat check failed! "
                                             r"The qkv in the model weights has been concatenated"):
            validate_qkv_concat(self.model, qkv_concat_config, self.weight_path_concat)

    def test_qkv_concat_with_not_supported_model(self):
        """test validate qkv concat config which is false and weights with concatenate"""
        model = ChatGLM2Model
        qkv_concat_config = self.config_concat.qkv_concat
        validate_qkv_concat(model, qkv_concat_config, self.weight_path_concat)

        log_content = self._get_log_content()
        self._reset_log_content()

        assert "does not support qkv concat check" in log_content

    def _get_log_content(self):
        with open(self.log_file_path, 'r') as log_file:
            log_content = log_file.read()
        return log_content

    def _reset_log_content(self):
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(self.log_file_path, flags_, 0o750), 'w') as log_file:
            log_file.write('')
