# Copyright 2023 Huawei Technologies Co., Ltd
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
"""InternLM transformer Layer's APIs."""
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode

from mindformers.models.llama.llama_transformer import LLamaAttention, LLamaDecodeLayer
from mindformers.modules.layers import Linear
from mindformers.modules.transformer import TransformerOpParallelConfig


class InternLMAttention(LLamaAttention):
    """Multi-head attention of InternLM inherited from LLamaAttention.

    Args:
        o_has_bias (bool, optional): Whether O projection in attention has bias. Defaults to True.
        **kwargs: keyword arguments of [`LLamaAttention`].

    """

    def __init__(self,
                 has_bias,
                 **kwargs):
        super().__init__(**kwargs)
        if has_bias:
            compute_dtype = kwargs.pop("compute_dtype", mstype.float16)
            param_init_type = kwargs.pop("param_init_type", mstype.float32)
            is_dynamic = kwargs.pop("is_dynamic", False)
            parallel_config = kwargs.pop("parallel_config", TransformerOpParallelConfig())

            self.wo = Linear(in_channels=self.hidden_size,
                             out_channels=self.hidden_size,
                             has_bias=has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wk = Linear(self.hidden_size,
                             self.n_kv_head * self.head_dim,
                             has_bias=has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wv = Linear(self.hidden_size,
                             self.n_kv_head * self.head_dim,
                             has_bias=has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)

            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
            if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
                self.wq.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
                self.wk.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
                self.wv.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
                self.wo.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))
                if parallel_config.use_seq_parallel and self.is_first_iteration:
                    self.wo.shard(((dp, mp), (1, mp)),
                                  out_strategy_matmul=((dp * mp, 1),),
                                  strategy_bias=((dp * mp, 1), (1,)))


class InternLMDecodeLayer(LLamaDecodeLayer):
    """InternLM Transformer Layer inherits from LLamaDecodeLayer.

    Args:
        layer_id (int): The layer id of current transformer block layer.
        o_has_bias (bool, optional): Whether O projection in attention has bias. Defaults to True.
        **kwargs: keyword arguments of [`LLamaDecodeLayer`].

    """

    def __init__(self,
                 layer_id,
                 has_bias,
                 **kwargs):
        super().__init__(layer_id=layer_id,
                         **kwargs)
        kwargs.pop("multiple_of")
        kwargs.pop("intermediate_size")
        kwargs.pop("ffn_dim_multiplier")
        kwargs.pop("norm_eps")
        kwargs.pop("layernorm_compute_dtype")
        self.attention = InternLMAttention(has_bias=has_bias, **kwargs)
