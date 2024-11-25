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
Note: p-tuning-v2 adapter
Reference: https://arxiv.org/pdf/2110.07602.pdf
"""
from typing import Union
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore._checkparam import args_type_check
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
try:
    from mindspore._checkparam import Rel, Validator
    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator
    INC_LEFT = Validator.INC_LEFT

from mindpet.utils.version_control import get_dropout

from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.pet_config import Ptuning2Config


class Ptuning2Embedding(nn.Cell):
    """
    The embedding with parallel_config.
    """

    def __init__(self, vocab_size, embedding_size, data_parallel=1, model_parallel=1, vocab_emb_dp=True,
                 param_init='normal', param_init_type=ms.float16):
        """
        vocab_size: vocab size
        embedding_size: embedding size
        data_parallel: data parallel config
        model_parallel: data parallel config
        vocab_emb_dp: embedding dp config
        param_init: parameter init method
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_table = ms.Parameter(initializer(param_init,
                                                        [self.vocab_size, self.embedding_size], dtype=param_init_type),
                                            name='embedding_table', parallel_optimizer=False)
        if vocab_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (data_parallel, 1)))
        else:
            if self.vocab_size % model_parallel != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of model_parallel {model_parallel}.")
            self.gather = P.Gather().shard(((model_parallel, 1), (data_parallel, 1)))

    def construct(self, input_ids):
        """
        embedding inputs
        """
        output = self.gather(self.embedding_table, input_ids, 0)
        return output


class Ptuning2Encoder(nn.Cell):
    """
    The cell to encode the prefix
    Input : batch_size
    Output shape: layers * (2, bs, pre_len, num_heads * kv_channels)
    """

    def __init__(
            self,
            pre_seq_len,
            num_layers,
            num_heads,
            kv_channels,
            prefix_projection,
            projection_dim,
            dropout_prob,
            parallel_config=None,
            out_perm=(2, 0, 1, 3),
    ):
        """
        Args:
            pre_seq_len: length of prefix
            num_layers: number of base model's transformer layers
            num_heads: number of base model's transformer attention heads
            kv_channels: dimension of of base model's transformer kv
            prefix_projection: Whether or not to use MLP projection
            projection_dim: MLP dimension
            dropout_prob: dropout rate
            parallel_config: parallel parameter
            out_perm: arrangement of output dimension
        """
        super().__init__()
        self.pre_seq_len = Validator.check_positive_int(pre_seq_len, "pre_seq_len")
        self.num_layers = Validator.check_positive_int(num_layers, "num_layers")
        self.num_heads = Validator.check_positive_int(num_heads, "num_heads")
        self.kv_channels = Validator.check_positive_int(kv_channels, "kv_channels")

        dropout_prob = Validator.check_float_range(dropout_prob, 0.0, 1.0, INC_LEFT)
        self.dropout = get_dropout(dropout_prob)

        self.prefix_projection = prefix_projection

        self.mindpet_delta_ptuning2_prefix = ms.Parameter(
            np.arange(self.pre_seq_len), requires_grad=False
        )

        out_embed_dim = self.num_layers * self.kv_channels * self.num_heads * 2
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        vocab_emb_dp = parallel_config.vocab_emb_dp

        self.mindpet_delta_ptuning2_embedding = Ptuning2Embedding(vocab_size=self.pre_seq_len,
                                                                  embedding_size=out_embed_dim,
                                                                  data_parallel=dp,
                                                                  model_parallel=mp,
                                                                  vocab_emb_dp=vocab_emb_dp)

        if self.prefix_projection:
            self.projection_dim = Validator.check_positive_int(
                projection_dim, "projection_dim"
            )
            # two-layer MLP to encode the prefix
            self.mindpet_delta_ptuning2_dense_in = nn.Dense(out_embed_dim, self.projection_dim, dtype=ms.float16)
            self.mindpet_delta_ptuning2_tanh = nn.Tanh()
            self.mindpet_delta_ptuning2_dense_out = nn.Dense(self.projection_dim, out_embed_dim, dtype=ms.float16)
            self.mindpet_delta_ptuning2_trans = nn.SequentialCell(
                self.mindpet_delta_ptuning2_dense_in,
                self.mindpet_delta_ptuning2_tanh,
                self.mindpet_delta_ptuning2_dense_out,
            )

        self.out_perm = out_perm
        self.expand_dims = P.ExpandDims().shard(((1,),))
        self.tile = P.Tile().shard(((1, 1),))
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.spilt_layers = P.Split(axis=0, output_num=self.num_layers)

    def construct(self, batch_size, dtype=mstype.half):
        """
        new prefix
        """
        prefix_tokens = self.expand_dims(self.mindpet_delta_ptuning2_prefix, 0)
        prefix_tokens = self.tile(prefix_tokens, (batch_size, 1))

        # (bs, pre_len) -> (bs, pre_len, 2 * layers * num_heads * kv_channels)
        past_key_values = self.mindpet_delta_ptuning2_embedding(prefix_tokens)

        if self.prefix_projection:
            past_key_values = self.mindpet_delta_ptuning2_trans(past_key_values)

        past_key_values = self.cast(past_key_values, dtype)

        # (bs, pre_len, 2 * layers * num_heads * kv_channels) -> (bs, pre_len, 2 * layers, num_heads * kv_channels)
        past_key_values = self.reshape(past_key_values, (batch_size,
                                                         self.pre_seq_len,
                                                         self.num_layers * 2,
                                                         self.num_heads * self.kv_channels
                                                         ))

        past_key_values = self.dropout(past_key_values)

        # (bs, pre_len, 2 * layers, num_heads * kv_channels) -> (2 * layers, bs, pre_len, num_heads * kv_channels)
        past_key_values = self.transpose(past_key_values, self.out_perm)

        # (2 * layers, bs, pre_len, num_heads * kv_channels) -> , layers * (2, bs, pre_len, num_heads * kv_channels)
        past_key_values = self.spilt_layers(past_key_values)

        return past_key_values

    def shard(self, data_parallel, model_parallel):
        """
        set shard strategy
        """

        if self.prefix_projection:
            # (bs, pre_len, embedding_dim)
            self.mindpet_delta_ptuning2_dense_in.matmul.shard(((data_parallel, 1), (model_parallel, 1)))
            self.mindpet_delta_ptuning2_dense_in.bias_add.shard(((data_parallel, 1), (1,)))
            self.mindpet_delta_ptuning2_tanh.tanh.shard(((data_parallel, 1, 1),))
            self.mindpet_delta_ptuning2_dense_out.matmul.shard(((data_parallel, 1), (model_parallel, 1)))
            self.mindpet_delta_ptuning2_dense_out.bias_add.shard(((data_parallel, 1), (1,)))

        # (bs, pre_len, 2 * layers * num_heads * kv_channels)
        self.cast.shard(((data_parallel, 1, 1),))  # (dp, 1, 1)

        # (bs, pre_len, 2 * layers, num_heads * kv_channels)
        self.dropout.dropout.shard(((data_parallel, 1, 1, 1),))
        self.transpose.shard(((data_parallel, 1, 1, 1),))

        # (2 * layers, bs, pre_len, num_heads * kv_channels)
        self.spilt_layers.shard(((1, data_parallel, 1, 1),))


class Ptuning2Adapter(PetAdapter):
    r"""
    Ptuning2 implement.
    """
    @classmethod
    @args_type_check(config=(dict, Ptuning2Config))
    def get_pet_model(cls, model: nn.Cell = None, config: Union[dict, Ptuning2Config] = None):
        pass

    @staticmethod
    def add_prefix(prefix_key_value, key, value, seq_len_dim=2):
        """
        Add p-tuning v2 prefix for key, vale. used by glmx
        """

        if prefix_key_value is not None:
            prefix_key = prefix_key_value[0]
            prefix_value = prefix_key_value[1]
            cat = P.Concat(seq_len_dim)
            prefix_key = P.Cast()(prefix_key, key.dtype)
            key = cat([prefix_key, key])
            prefix_value = P.Cast()(prefix_value, value.dtype)
            value = cat([prefix_value, value])

        return key, value

    @classmethod
    @args_type_check(config=(dict, Ptuning2Config))
    def get_prefix(cls, model: nn.Cell = None, config: Union[dict, Ptuning2Config] = None):
        """
        return prefix prompt.
        """
        if not isinstance(config, Ptuning2Config):
            config = config.copy()
            config.pop("pet_type")
            config = Ptuning2Config(**config)
        num_layers = model.config.num_layers
        num_heads = model.config.num_heads
        kv_channels = model.config.hidden_size // model.config.num_heads  # 128
        parallel_config = model.config.parallel_config
        prefix_projection = config["prefix_projection"]
        projection_dim = config["projection_dim"]
        dropout_prob = config["dropout_prob"]
        pre_seq_len = config["pre_seq_len"]

        prefixs = Ptuning2Encoder(pre_seq_len=pre_seq_len, num_layers=num_layers, num_heads=num_heads,
                                  kv_channels=kv_channels, prefix_projection=prefix_projection,
                                  projection_dim=projection_dim, dropout_prob=dropout_prob,
                                  parallel_config=parallel_config, out_perm=(2, 0, 1, 3))
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            prefixs.pipeline_stage = 0
            prefixs.shard(parallel_config.data_parallel, parallel_config.model_parallel)

        return prefixs
