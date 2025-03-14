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
"""Test Normalization"""
import pytest

import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.ops.operations as P

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from tests.st.test_static_distri_core.test_norm.test_norm_utils import MyNet
from tests.st.test_static_distri_core.test_norm.test_norm_utils import get_output

class TestNormalization:
    """A test class for testing LayerNorm/FusedLayerNorm/RMSNorm/FusedRMSNorm."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_run_norm_single(self):
        """
        Feature: get_norm()
        Description: Test get_norm on one card
        Expectation: AssertionError
        """
        seed = 22
        ms.set_seed(seed)
        np.random.seed(seed)

        batch, seq_length, hidden_size = (2, 3, 4)
        config = TransformerConfig()
        config.hidden_size = hidden_size
        config.layernorm_epsilon = 1e-6

        input_shape = (batch, seq_length, hidden_size)
        input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        data_type_list = [ms.float16, ms.float32, ms.bfloat16]
        cast = P.Cast()
        for data_type in data_type_list:
            config.layernorm_compute_type = data_type
            mynet = MyNet(config)
            input_ = cast(input_, data_type)
            output_0, output_1, output_2, output_3 = mynet(input_)
            expected_output_0, expected_output_1 = get_output()
            if data_type == ms.float32:
                assert np.allclose(output_0.asnumpy(), expected_output_0)
                assert np.allclose(output_1.asnumpy(), expected_output_0)
                assert np.allclose(output_2.asnumpy(), expected_output_1)
                assert np.allclose(output_3.asnumpy(), expected_output_1)
