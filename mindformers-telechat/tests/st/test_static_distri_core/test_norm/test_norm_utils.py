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
"""run norm utils"""
import numpy as np

import mindspore.nn as nn

from mindformers.experimental.graph.transformer.norm import get_norm

class MyNet(nn.Cell):
    """ test net """
    def __init__(self, config):
        super(MyNet, self).__init__()

        config.normalization = 'LayerNorm'
        self.layernorm = get_norm(config)
        self.layernorm.shard(config)

        config.normalization = 'FusedLayerNorm'
        self.fused_layernorm = get_norm(config)
        self.fused_layernorm.shard(config)

        config.normalization = 'RMSNorm'
        self.rmsnorm = get_norm(config)
        self.rmsnorm.shard(config)

        config.normalization = 'FusedRMSNorm'
        self.fused_rmsnorm = get_norm(config)
        self.fused_rmsnorm.shard(config)

    def construct(self, x):
        """ forward """
        out0 = self.layernorm(x)
        out1 = self.fused_layernorm(x)
        out2 = self.rmsnorm(x)
        out3 = self.fused_rmsnorm(x)
        return out0, out1, out2, out3

def get_output():
    """ outputs from megatron-gpu """
    output_0 = np.array([[[0.09549448, -1.4254426, 1.3979844, -0.06803622],
                          [-0.08896536, -0.72201145, 1.6588713, -0.84789443],
                          [1.518591, -1.2119741, 0.14499639, -0.4516133]],
                         [[0.30861774, 0.5089147, -1.6955403, 0.8780078],
                          [0.24730077, 0.5208384, 0.91449654, -1.6826357],
                          [1.4632763, -1.240388, -0.49577358, 0.27288538]]])
    output_1 = np.array([[[-0.10011128, -1.5922143, 1.1776859, -0.26054174],
                          [-0.54019934, -1.1018778, 1.0105916, -1.2135692],
                          [1.43548, -1.2866734, 0.06611683, -0.52865493]],
                         [[0.66281074, 0.8483317, -1.1934998, 1.1901968],
                          [0.5995436, 0.85369366, 1.2194505, -1.1936047],
                          [1.4540553, -1.2494953, -0.50491214, 0.2637145]]])
    return output_0, output_1
