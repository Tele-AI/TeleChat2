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
# This file was refer to project:
# https://github.com/facebookresearch/mae
# ============================================================================
"""EVA-02 MLPs' APIs."""
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.modules.layers import Linear


class Mlp(nn.Cell):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_size,
                 hidden_size=None,
                 out_size=None,
                 norm_layer=None,
                 bias=True,
                 drop_prob=0.,
                 conv_linear=False,
                 layer_norm_eps=1e-6,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 layer_norm_type=mstype.float32):
        super().__init__()
        out_size = out_size or in_size
        hidden_size = hidden_size or in_size
        bias = (bias, bias)
        drop_probs = (drop_prob, drop_prob)

        if conv_linear:
            self.fc1 = nn.Conv2d(in_size, hidden_size, has_bias=bias[0], dtype=param_init_type)
            self.fc2 = nn.Conv2d(hidden_size, out_size, bias=bias[1], dtype=param_init_type)
        else:
            self.fc1 = Linear(in_size, hidden_size, has_bias=bias[0],
                              param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.fc2 = Linear(hidden_size, out_size, has_bias=bias[1],
                              param_init_type=param_init_type, compute_dtype=compute_dtype)

        self.act = nn.GELU(approximate=False)
        self.drop1 = nn.Dropout(p=drop_probs[0], dtype=param_init_type)
        if norm_layer:
            self.norm = nn.LayerNorm((hidden_size,), epsilon=layer_norm_eps, dtype=layer_norm_type)
        else:
            self.norm = nn.Identity()
        self.drop2 = nn.Dropout(p=drop_probs[1], dtype=param_init_type)

    def construct(self, x):
        """Mlp Forward."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Cell):
    """
    MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(self,
                 in_size,
                 hidden_size=None,
                 out_size=None,
                 act_layer=nn.Sigmoid,
                 norm_layer=None,
                 bias=True,
                 drop_prob=0.,
                 conv_linear=False,
                 gate_last=True,
                 layer_norm_eps=1e-6,
                 compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 layer_norm_type=mstype.float32):
        super().__init__()
        out_size = out_size or in_size
        hidden_size = hidden_size or in_size
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be an even number.")

        bias = (bias, bias)
        drop_probs = (drop_prob, drop_prob)

        if conv_linear:
            self.fc1 = nn.Conv2d(in_size, hidden_size, has_bias=bias[0], dtype=compute_dtype)
            self.fc2 = nn.Conv2d(hidden_size // 2, out_size, has_bias=bias[1], dtype=compute_dtype)
        else:
            self.fc1 = Linear(in_size, hidden_size, has_bias=bias[0],
                              param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.fc2 = Linear(hidden_size // 2, out_size, has_bias=bias[1],
                              param_init_type=param_init_type, compute_dtype=compute_dtype)

        self.chunk_dim = 1 if conv_linear else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.act = act_layer()
        if norm_layer:
            self.norm = nn.LayerNorm((hidden_size // 2,), epsilon=layer_norm_eps, dtype=layer_norm_type)
        else:
            self.norm = nn.Identity()
        self.drop1 = nn.Dropout(p=drop_probs[0], dtype=compute_dtype)
        self.drop2 = nn.Dropout(p=drop_probs[1], dtype=compute_dtype)

        self.mul = P.Mul()

    def construct(self, x):
        """GluMlp Forward."""
        x = self.fc1(x)
        x1, x2 = F.chunk(x, 2, axis=self.chunk_dim)
        if self.gate_last:
            x = self.mul(x1, self.act(x2))
        else:
            x = self.mul(self.act(x1), x2)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLU(nn.Cell):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """

    def __init__(self,
                 in_size,
                 hidden_size=None,
                 out_size=None,
                 act_layer=nn.SiLU,
                 norm_layer=True,
                 bias=True,
                 drop_prob=0.,
                 layer_norm_eps=1e-6,
                 compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 layer_norm_type=mstype.float32):
        super().__init__()
        out_size = out_size or in_size
        hidden_size = hidden_size or in_size
        bias = (bias, bias)
        drop_probs = (drop_prob, drop_prob)

        self.fc1_g = Linear(in_size, hidden_size, has_bias=bias[0],
                            param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.fc1_x = Linear(in_size, hidden_size, has_bias=bias[0],
                            param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0], dtype=compute_dtype)
        if norm_layer:
            self.norm = nn.LayerNorm((hidden_size,), epsilon=layer_norm_eps, dtype=layer_norm_type)
        else:
            self.norm = nn.Identity()

        self.fc2 = Linear(hidden_size, out_size, has_bias=bias[1],
                          param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.drop2 = nn.Dropout(p=drop_probs[1], dtype=compute_dtype)

        self.mul = P.Mul()

    def construct(self, x):
        """SwiGLU Forward."""
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.mul(self.act(x_gate), x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
