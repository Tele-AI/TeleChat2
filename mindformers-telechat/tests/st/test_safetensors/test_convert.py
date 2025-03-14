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
Test module for testing the convert_hf_safetensors_multiprocess interface used for mindformers.
How to run this:
pytest tests/st/test_safetensors/test_convert.py
"""
import shutil
import json
import os
import tempfile

import pytest

import numpy as np
from safetensors.numpy import save_file, load_file

from mindformers import LlamaForCausalLM
from mindformers.tools import MindFormerConfig
from mindformers.utils import (
    convert_hf_safetensors_multiprocess,
    contains_safetensors_files,
    is_hf_safetensors_dir,
    qkv_concat_hf2mg,
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestConvert:
    """A test class for testing convert safetensors."""

    def setup_class(self):
        """set fake model attention config"""
        self.num_heads = 2
        self.n_kv_heads = 1
        self.hidden_size = 2
        self.n_rep = self.num_heads // self.n_kv_heads

        # create fake safetensors directory
        work_dir = tempfile.mkdtemp()
        src_path_unified = os.path.join(work_dir, "hf_path_unified")
        os.makedirs(src_path_unified)
        dst_path_unified = os.path.join(work_dir, "ms_path_unified")
        src_path_single = os.path.join(work_dir, "hf_path_single")
        os.makedirs(src_path_single)
        dst_path_single = os.path.join(work_dir, "ms_path_single")

        # create fake weights value
        self.fake_weight1 = np.random.rand(self.hidden_size, self.hidden_size)
        self.fake_weight2 = np.random.rand(self.hidden_size // self.n_rep, self.hidden_size)
        self.fake_weight3 = np.random.rand(self.hidden_size // self.n_rep, self.hidden_size)

        # 1. prepare unified safetensors dir
        # prepare fake weight dict and map
        fake_dict_1 = {"model.layers.0.self_attn.q_proj.weight": self.fake_weight1}
        fake_dict_2 = {"model.layers.0.self_attn.k_proj.weight": self.fake_weight2}
        fake_dict_3 = {"model.layers.0.self_attn.v_proj.weight": self.fake_weight3}
        fake_map = {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
                                   "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00003.safetensors",
                                   "model.layers.0.self_attn.v_proj.weight": "model-00003-of-00003.safetensors"}}
        fake_file_1 = os.path.join(src_path_unified, "model-00001-of-00003.safetensors")
        fake_file_2 = os.path.join(src_path_unified, "model-00002-of-00003.safetensors")
        fake_file_3 = os.path.join(src_path_unified, "model-00003-of-00003.safetensors")
        fake_map_file = os.path.join(src_path_unified, 'model.safetensors.index.json')

        # make fake safetensors file
        save_file(tensor_dict=fake_dict_1, filename=fake_file_1)
        os.chmod(fake_file_1, 0o750)
        save_file(tensor_dict=fake_dict_2, filename=fake_file_2)
        os.chmod(fake_file_2, 0o750)
        save_file(tensor_dict=fake_dict_3, filename=fake_file_3)
        os.chmod(fake_file_3, 0o750)

        # make fake json file
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(fake_map_file, flags_, 0o750), 'w') as f:
            json.dump(fake_map, f)

        # 2. prepare single safetensors dir
        # prepare fake weight dict
        fake_dict = {"model.layers.0.self_attn.q_proj.weight": self.fake_weight1,
                     "model.layers.0.self_attn.k_proj.weight": self.fake_weight2,
                     "model.layers.0.self_attn.v_proj.weight": self.fake_weight3}
        fake_file = os.path.join(src_path_single, "model.safetensors")

        # make fake safetensors file
        save_file(tensor_dict=fake_dict, filename=fake_file)
        os.chmod(fake_file, 0o750)

        self.work_dir = work_dir
        self.src_path_unified = src_path_unified
        self.dst_path_unified = dst_path_unified
        self.src_path_single = src_path_single
        self.dst_path_single = dst_path_single
        model_dict = {'model':
                          {'model_config':
                               {'num_heads': self.num_heads,
                                'n_kv_heads': self.n_kv_heads,
                                'hidden_size': self.hidden_size,
                                'qkv_concat': False}
                           }
                      }
        self.model_config = MindFormerConfig(**model_dict).model.model_config

    def teardown_class(self):
        """remove fake safetensors directory"""
        shutil.rmtree(self.work_dir)

    @pytest.mark.run(order=1)
    def test_convert_unified(self):
        """test convert unified HuggingFace safetensors to MindSpore safetensors"""
        self.model_config.qkv_concat = False
        convert_hf_safetensors_multiprocess(self.src_path_unified,
                                            self.dst_path_unified,
                                            LlamaForCausalLM,
                                            self.model_config)

        converted_file_1 = os.path.join(self.dst_path_unified, "model-00001-of-00003.safetensors")
        converted_file_2 = os.path.join(self.dst_path_unified, "model-00002-of-00003.safetensors")
        converted_file_3 = os.path.join(self.dst_path_unified, "model-00003-of-00003.safetensors")
        converted_map_file = os.path.join(self.dst_path_unified, "param_name_map.json")

        # check whether safetensors and map file exists
        assert os.path.exists(converted_file_1), \
            "model-00001-of-00003.safetensors does not exist, conversion not completed."
        assert os.path.exists(converted_file_2), \
            "model-00002-of-00003.safetensors does not exist, conversion not completed."
        assert os.path.exists(converted_file_3), \
            "model-00003-of-00003.safetensors does not exist, conversion not completed."
        assert os.path.exists(converted_map_file), \
            "param_name_map.json does not exist, conversion not completed."

        # check whether the weight in the safetensors file is correct
        converted_dict_1 = load_file(converted_file_1)
        assert "model.layers.0.attention.wq.weight" in converted_dict_1.keys(), \
            "model-00001-of-00003.safetensors does not have key 'model.layers.0.attention.wq.weight'."
        assert np.array_equal(converted_dict_1.get("model.layers.0.attention.wq.weight"), self.fake_weight1), \
            f"The value of 'model.layers.0.attention.wq.weight' should be {self.fake_weight1}, " \
            f"but got {converted_dict_1.get('model.layers.0.attention.wq.weight')}."
        converted_dict_2 = load_file(converted_file_2)
        assert "model.layers.0.attention.wk.weight" in converted_dict_2.keys(), \
            "model-00002-of-00003.safetensors does not have key 'model.layers.0.attention.wk.weight'."
        assert np.array_equal(converted_dict_2.get("model.layers.0.attention.wk.weight"), self.fake_weight2), \
            f"The value of 'model.layers.0.attention.wk.weight' should be {self.fake_weight2}, " \
            f"but got {converted_dict_2.get('model.layers.0.attention.wk.weight')}."
        converted_dict_3 = load_file(converted_file_3)
        assert "model.layers.0.attention.wv.weight" in converted_dict_3.keys(), \
            "model-00003-of-00003.safetensors does not have key 'model.layers.0.attention.wv.weight'."
        assert np.array_equal(converted_dict_3.get("model.layers.0.attention.wv.weight"), self.fake_weight3), \
            f"The value of 'model.layers.0.attention.wv.weight' should be {self.fake_weight3}, " \
            f"but got {converted_dict_3.get('model.layers.0.attention.wv.weight')}."

        # check whether the map in the json file is correct
        with open(converted_map_file, 'r') as f:
            converted_map = json.load(f)
        assert "model.layers.0.attention.wq.weight" in converted_map.keys(), \
            "param_name_map.json does not have key 'model.layers.0.attention.wq.weight'."
        assert converted_map.get("model.layers.0.attention.wq.weight") == "model-00001-of-00003.safetensors", \
            f"The value of key 'model.layers.0.attention.wq.weight' should be 'model-00001-of-00003.safetensors' " \
            f"in the param_name_map.json, but got {converted_map.get('model.layers.0.attention.wq.weight')}."
        assert "model.layers.0.attention.wk.weight" in converted_map.keys(), \
            "param_name_map.json does not have key 'model.layers.0.attention.wk.weight'."
        assert converted_map.get("model.layers.0.attention.wk.weight") == "model-00002-of-00003.safetensors", \
            f"The value of key 'model.layers.0.attention.wk.weight' should be 'model-00002-of-00003.safetensors' " \
            f"in the param_name_map.json, but got {converted_map.get('model.layers.0.attention.wk.weight')}."
        assert "model.layers.0.attention.wv.weight" in converted_map.keys(), \
            "param_name_map.json does not have key 'model.layers.0.attention.wv.weight'."
        assert converted_map.get("model.layers.0.attention.wv.weight") == "model-00003-of-00003.safetensors", \
            f"The value of key 'model.layers.0.attention.wv.weight' should be 'model-00003-of-00003.safetensors' " \
            f"in the param_name_map.json, but got {converted_map.get('model.layers.0.attention.wv.weight')}."

    @pytest.mark.run(order=2)
    def test_convert_unified_with_qkv_concat(self):
        """test convert HuggingFace safetensors to MindSpore safetensors with concatenating q,k,v weight"""
        self.model_config.qkv_concat = True
        convert_hf_safetensors_multiprocess(self.src_path_unified,
                                            self.dst_path_unified,
                                            LlamaForCausalLM,
                                            self.model_config)

        converted_file = os.path.join(self.dst_path_unified, "model-00001-of-00003.safetensors")
        converted_map_file = os.path.join(self.dst_path_unified, "param_name_map.json")

        # check whether safetensors and map file exists
        assert os.path.exists(converted_file), \
            "model-00001-of-00003.safetensors does not exist, conversion not completed."
        assert os.path.exists(converted_map_file), \
            "param_name_map.json does not exist, conversion not completed."

        # check whether the weight in the safetensors file is correct
        converted_dict = load_file(converted_file)
        assert "model.layers.0.attention.w_qkv.weight" in converted_dict.keys(), \
            "model-00001-of-00003.safetensors does not have key 'model.layers.0.attention.w_qkv.weight'."

        expect_qkv_weights = qkv_concat_hf2mg(np.concatenate(self.fake_weight1, self.fake_weight2, self.fake_weight3),
                                              self.num_heads, self.n_kv_heads, self.hidden_size)
        assert np.array_equal(converted_dict.get("model.layers.0.attention.w_qkv.weight"), expect_qkv_weights), \
            f"The value of 'model.layers.0.attention.w_qkv.weight' got \
            {converted_dict.get('model.layers.0.attention.w_qkv.weight')}, " \
            f"not the same as expected weights {expect_qkv_weights}."

        # check whether the map in the json file is correct
        with open(converted_map_file, 'r') as f:
            converted_map = json.load(f)
        assert "model.layers.0.attention.w_qkv.weight" in converted_map.keys(), \
            "param_name_map.json does not have key 'model.layers.0.attention.w_qkv.weight'."
        assert converted_map.get("model.layers.0.attention.w_qkv.weight") == "model-00001-of-00003.safetensors", \
            f"The value of key 'model.layers.0.attention.w_qkv.weight' " \
            f"should be 'model-00001-of-00003.safetensors' " \
            f"in the param_name_map.json, but got {converted_map.get('model.layers.0.attention.w_qkv.weight')}."

    @pytest.mark.run(order=3)
    def test_convert_single(self):
        """test convert single HuggingFace safetensors to MindSpore safetensors"""
        self.model_config.qkv_concat = False
        convert_hf_safetensors_multiprocess(self.src_path_single,
                                            self.dst_path_single,
                                            LlamaForCausalLM,
                                            self.model_config)

        converted_file = os.path.join(self.dst_path_single, "model.safetensors")

        # check whether safetensors file exists
        assert os.path.exists(converted_file), \
            "model-00001-of-00003.safetensors does not exist, conversion not completed."

        # check whether the weight in the safetensors file is correct
        converted_dict = load_file(converted_file)
        assert "model.layers.0.attention.wq.weight" in converted_dict.keys(), \
            "model.safetensors does not have key 'model.layers.0.attention.wq.weight'."
        assert np.array_equal(converted_dict.get("model.layers.0.attention.wq.weight"), self.fake_weight1), \
            f"The value of 'model.layers.0.attention.wq.weight' should be {self.fake_weight1}, " \
            f"but got {converted_dict.get('model.layers.0.attention.wq.weight')}."
        assert "model.layers.0.attention.wk.weight" in converted_dict.keys(), \
            "model.safetensors does not have key 'model.layers.0.attention.wk.weight'."
        assert np.array_equal(converted_dict.get("model.layers.0.attention.wk.weight"), self.fake_weight2), \
            f"The value of 'model.layers.0.attention.wk.weight' should be {self.fake_weight2}, " \
            f"but got {converted_dict.get('model.layers.0.attention.wk.weight')}."
        assert "model.layers.0.attention.wv.weight" in converted_dict.keys(), \
            "model.safetensors does not have key 'model.layers.0.attention.wv.weight'."
        assert np.array_equal(converted_dict.get("model.layers.0.attention.wv.weight"), self.fake_weight3), \
            f"The value of 'model.layers.0.attention.wv.weight' should be {self.fake_weight3}, " \
            f"but got {converted_dict.get('model.layers.0.attention.wv.weight')}."

    @pytest.mark.run(order=4)
    def test_convert_single_with_qkv_concat(self):
        """test convert single HuggingFace safetensors to MindSpore safetensors with concatenating q,k,v weight"""
        self.model_config.qkv_concat = True
        convert_hf_safetensors_multiprocess(self.src_path_single,
                                            self.dst_path_single,
                                            LlamaForCausalLM,
                                            self.model_config)

        converted_file = os.path.join(self.dst_path_single, "model.safetensors")

        # check whether safetensors and map file exists
        assert os.path.exists(converted_file), \
            "model.safetensors does not exist, conversion not completed."

        # check whether the weight in the safetensors file is correct
        converted_dict = load_file(converted_file)
        assert "model.layers.0.attention.w_qkv.weight" in converted_dict.keys(), \
            "model.safetensors does not have key 'model.layers.0.attention.w_qkv.weight'"

        expect_qkv_weights = qkv_concat_hf2mg(np.concatenate(self.fake_weight1, self.fake_weight2, self.fake_weight3),
                                              self.num_heads, self.n_kv_heads, self.hidden_size)
        assert np.array_equal(converted_dict.get("model.layers.0.attention.w_qkv.weight"), expect_qkv_weights), \
            f"The value of 'model.layers.0.attention.w_qkv.weight' got \
            {converted_dict.get('model.layers.0.attention.w_qkv.weight')}, " \
            f"not the same as expected weights {expect_qkv_weights}."

    @pytest.mark.run(order=5)
    def test_convert_input_error(self):
        """test error input of convert_hf_safetensors_multiprocess"""
        self.model_config.qkv_concat = False
        with pytest.raises(ValueError):
            convert_hf_safetensors_multiprocess(1,
                                                self.dst_path_single,
                                                LlamaForCausalLM,
                                                self.model_config)
        with pytest.raises(ValueError):
            convert_hf_safetensors_multiprocess(self.src_path_single,
                                                1,
                                                LlamaForCausalLM,
                                                self.model_config)
        with pytest.raises(ValueError):
            convert_hf_safetensors_multiprocess(self.src_path_single,
                                                self.dst_path_single,
                                                1,
                                                self.model_config)
        with pytest.raises(ValueError):
            convert_hf_safetensors_multiprocess(self.work_dir,
                                                self.dst_path_single,
                                                LlamaForCausalLM,
                                                self.model_config)
        with pytest.raises(ValueError):
            self.model_config.qkv_concat = 1
            convert_hf_safetensors_multiprocess(self.src_path_single,
                                                self.dst_path_single,
                                                LlamaForCausalLM,
                                                self.model_config)

    @pytest.mark.run(order=6)
    def test_utils(self):
        """test utilities of safetensors"""
        res = contains_safetensors_files(self.src_path_unified)
        assert res, f"contains_safetensors_files should return True, but got {res}."

        res = contains_safetensors_files(self.work_dir)
        assert not res, f"contains_safetensors_files should return False, but got {res}."

        res = is_hf_safetensors_dir(self.src_path_unified, LlamaForCausalLM)
        assert res, f"is_hf_safetensors_dir should return True, but got {res}."

        res = is_hf_safetensors_dir(self.src_path_single, LlamaForCausalLM)
        assert res, f"is_hf_safetensors_dir should return True, but got {res}."

        res = is_hf_safetensors_dir(self.work_dir, LlamaForCausalLM)
        assert not res, f"is_hf_safetensors_dir should return False, but got {res}."

        res = is_hf_safetensors_dir(self.dst_path_unified, LlamaForCausalLM)
        assert not res, f"is_hf_safetensors_dir should return False, but got {res}."

        res = is_hf_safetensors_dir(self.dst_path_single, LlamaForCausalLM)
        assert not res, f"is_hf_safetensors_dir should return False, but got {res}."
