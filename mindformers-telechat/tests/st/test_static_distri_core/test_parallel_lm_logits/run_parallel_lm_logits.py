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
"""Run vocab parallel embedding test"""
import argparse
import os
import inspect
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.communication import init
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.transformer import ParallelLMLogits


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dp',
    default=1,
    required=True,
    type=int,
    help='data_parallel')
parser.add_argument(
    '--cp',
    default=1,
    required=True,
    type=int,
    help='context_parallel')
parser.add_argument(
    '--tp',
    default=1,
    required=True,
    type=int,
    help='tensor_parallel')
args_, rest_args_ = parser.parse_known_args()

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


class ParallelLMLogitsNet(nn.Cell):
    """ParallelLMLogitsNet for test vocab parallel embedding"""
    def __init__(self, config: TransformerConfig, bias: bool):
        super(ParallelLMLogitsNet, self).__init__()
        self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                   bias=bias,
                                                   compute_dtype=mstype.float16)

        module_path = inspect.getsourcefile(ParallelLMLogits)
        print(f'module_path:{module_path}')
        self.mul = P.Mul()
        self.mul.shard(((1, 1, 1), (1, 1)))
        self.rank_id = os.environ.get('RANK_ID')
        print(f"rank_id:{self.rank_id}")
        if self.rank_id is None:
            self.rank_id = 0

    def construct(self,
                  lm_output: Tensor,
                  logit_weights: Tensor,
                  parallel_output: bool,
                  bias: Tensor):
        output_ = self.parallel_lm_logits(lm_output, logit_weights, parallel_output, bias)
        return output_


if __name__ == "__main__":
    rank_id = os.environ.get('RANK_ID')

    config_ = TransformerConfig()
    config_.data_parallel = args_.dp
    config_.tensor_parallel = args_.tp
    config_.context_parallel = args_.cp


    net = ParallelLMLogitsNet(config_, bias=True)

    input_ = Tensor(np.array([[0.5317, 0.8311, 0.9717, 0.1193],
                              [0.1669, 0.3494, 0.2150, 0.6201],
                              [0.4849, 0.7490, 0.1521, 0.5625],
                              [0.1735, 0.7046, 0.9268, 0.4060]], dtype=np.float16))

    word_embedding_weight = Tensor(np.array([[0.3044, 0.9072, 0.7637, 0.6104],
                                             [0.8916, 0.9092, 0.6650, 0.6787],
                                             [0.2268, 0.5029, 0.2151, 0.7988],
                                             [0.0784, 0.0706, 0.2971, 0.0692],
                                             [0.9829, 0.6968, 0.6904, 0.4407],
                                             [0.5645, 0.1914, 0.3562, 0.5625],
                                             [0.0469, 0.4519, 0.4587, 0.7573],
                                             [0.9927, 0.5146, 0.4524, 0.9702]], dtype=np.float16))

    bias_ = Tensor(np.array([0.2343, 0.6592, 0.6729, 0.1998, 0.2683, 0.5767, 0.2734, 0.4980], dtype=np.float16))

    parallel_output_ = True

    logits = net(lm_output=input_, logit_weights=word_embedding_weight, parallel_output=parallel_output_, bias=bias_)

    pta_res = 1.482421875
    error = 1e-6
    assert abs(logits.mean() - pta_res) < error
