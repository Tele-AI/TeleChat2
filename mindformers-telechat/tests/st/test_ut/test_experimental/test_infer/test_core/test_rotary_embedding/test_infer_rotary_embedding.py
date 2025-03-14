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
"""test rotary embedding in infer mode"""
import os
import pytest


class TestInferRotaryEmbedding:
    """A test class for testing RotaryEmbedding"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("max_position", [256])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32])
    @pytest.mark.parametrize("hidden_size", [32])
    @pytest.mark.parametrize("prefill", [0, 1])
    def test_rotary_embedding_on_single(self, max_position, batch_size, seq_len, hidden_size, prefill):
        """
        Feature: RotaryEmbedding
        Description: Test RotaryEmbedding
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        os.environ['MS_ENABLE_LCCL'] = "off"
        num_heads = 4
        ret = os.system(f"python {sh_path}/run_infer_rotary_embedding.py --mode rope --bs {batch_size} --seq {seq_len}"
                        f" --hidden {hidden_size} --num {num_heads} --position {max_position} --prefill {prefill}")
        assert ret == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("max_position", [256])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32])
    @pytest.mark.parametrize("hidden_size", [32])
    @pytest.mark.parametrize("prefill", [0, 1])
    def test_llama3_rotary_embedding_on_single(self, max_position, batch_size, seq_len, hidden_size, prefill):
        """
        Feature: Llama3RotaryEmbedding
        Description: Test Llama3RotaryEmbedding
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        os.environ['MS_ENABLE_LCCL'] = "off"
        num_heads = 8
        ret = os.system(f"python {sh_path}/run_infer_rotary_embedding.py --mode llama3rope --bs {batch_size} "
                        f"--seq {seq_len} --hidden {hidden_size} "
                        f"--num {num_heads} --position {max_position} --prefill {prefill}")
        assert ret == 0
