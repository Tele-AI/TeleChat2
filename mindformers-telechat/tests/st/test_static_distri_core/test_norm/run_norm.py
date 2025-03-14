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
"""run norm test"""
import argparse

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.communication import init
import mindspore.ops.operations as P

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from tests.st.test_static_distri_core.test_norm.test_norm_utils import MyNet

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
init()

seed = 22
ms.set_seed(seed)
np.random.seed(seed)

def run_main():
    """ test norm """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=1, required=True, type=int, help='data_parallel')
    parser.add_argument('--cp', default=1, required=True, type=int, help='context_parallel')
    parser.add_argument('--tp', default=1, required=True, type=int, help='tensor_parallel')
    args_, rest_args_ = parser.parse_known_args()
    print("args:", args_)
    print("rest_args:", rest_args_)

    batch, seq_length, hidden_size = (32, 1024, 256)
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
        out0, out1, out2, out3 = mynet(input_)
        print(f"{out0.shape}, {out1.shape}, {out2.shape}, {out3.shape}")
        print(f"test data_type {data_type} complete!")


run_main()
