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
"""run mlp in infer mode"""
import numpy as np

import mindspore as ms
from mindspore import Tensor

from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.models.llama import LlamaConfig
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

from tests.st.test_ut.test_experimental.test_infer.test_core.test_mlp.utils import (
    NewMLPNet,
    OldMLPNet,
    convert_weight_name,
)


jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, full_batch=False)

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


def set_model_config():
    """Set model config for mlp ut test."""
    config = LlamaConfig()
    config.batch_size = 1
    config.seq_length = 32
    config.hidden_size = 64
    config.intermediate_size = 128
    config.hidden_act = 'silu'
    config.mlp_has_bias = False
    config.mlp_has_gate = True
    config.ffn_concat = False
    return config


if __name__ == '__main__':
    model_config = set_model_config()
    transformer_config = TransformerConfig()
    transformer_config = convert_to_transformer_config(model_config, transformer_config)

    bs = model_config.batch_size
    seq_len = model_config.seq_length
    hidden_size = model_config.hidden_size

    input_shape = (1, bs * seq_len, hidden_size)
    input_x = Tensor(np.random.standard_normal(input_shape).astype(np.float16))

    new_net = NewMLPNet(config=transformer_config)
    old_net = OldMLPNet(config=model_config)
    old_param_dict = old_net.parameters_dict()

    converted_param_dict = convert_weight_name(old_param_dict)
    ms.load_param_into_net(new_net, converted_param_dict)

    new_mlp_output = new_net(input_x)
    old_mlp_output = old_net(input_x)

    ret = np.array_equal(new_mlp_output.asnumpy(), old_mlp_output.asnumpy())
    assert ret
