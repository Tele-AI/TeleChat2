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
"""test infer rotary embedding utils"""
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.modules.layers import FreqsMgr, SeqExtendMethod
from mindformers.modules.infer_attention import InferRotaryEmbedding
from mindformers.experimental.infer.core.rotary_embedding import RotaryEmbedding, Llama3RotaryEmbedding


class NewRopeNet(nn.Cell):
    """A model class of new rotary embedding."""
    def __init__(self, config: TransformerConfig, rotary_cos_format=0, prefill: bool = True):
        super(NewRopeNet, self).__init__()
        if config.seq_length > config.max_position_embeddings:
            self.max_position_embeddings = config.seq_length
        else:
            self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.rotary_cos_format = rotary_cos_format
        self.prefill = prefill
        self.rotary_embedding = RotaryEmbedding(kv_channels=self.head_dim,
                                                rotary_cos_format=self.rotary_cos_format)

    def construct(self, query: Tensor, key: Tensor, batch_valid_length):
        if self.prefill:
            freqs_cos, freqs_sin = self.rotary_embedding.get_cos_sin_for_prefill(self.max_position_embeddings)
        else:
            indices = batch_valid_length - 1
            freqs_cos, freqs_sin = \
                self.rotary_embedding.get_cos_sin_for_decode(indices,
                                                             self.max_position_embeddings)
        return self.rotary_embedding(query, key, freqs_cos, freqs_sin, batch_valid_length)


class OldRopeNet(nn.Cell):
    """A model class of old rotary embedding."""
    def __init__(self, config: TransformerConfig, rotary_cos_format=0, prefill: bool = True):
        super(OldRopeNet, self).__init__()
        self.seq_len = config.seq_length
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=self.seq_len,
                                  max_position_embedding=self.max_position_embeddings,
                                  rotary_dtype=mstype.float16,
                                  is_dynamic=True)
        self.rotary_cos_format = rotary_cos_format
        self.prefill = prefill
        self.rotary_embedding = InferRotaryEmbedding(self.rotary_cos_format)

    def construct(self, query: Tensor, key: Tensor, batch_valid_length):
        if self.prefill:
            freqs_cis = self.freqs_mgr.prefill(query.shape[0], self.seq_len)
        else:
            freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        return self.rotary_embedding(query, key, freqs_cis, batch_valid_length)


class NewLlama3RopeNet(nn.Cell):
    """A model class of new llama3 rotary embedding."""
    def __init__(self, config: TransformerConfig, rotary_cos_format=0, prefill: bool = True):
        super(NewLlama3RopeNet, self).__init__()
        self.seq_len = config.seq_length
        if config.seq_length > config.max_position_embeddings:
            self.max_position_embeddings = config.seq_length
        else:
            self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads

        self.rotary_cos_format = rotary_cos_format
        self.prefill = prefill
        self.rotary_embedding = Llama3RotaryEmbedding(kv_channels=self.head_dim,
                                                      rotary_base=500000,
                                                      rotary_cos_format=self.rotary_cos_format,
                                                      scaling_factor=8.0,
                                                      orig_max_position=128)

    def construct(self, query: Tensor, key: Tensor, batch_valid_length):
        if self.prefill:
            freqs_cos, freqs_sin = self.rotary_embedding.get_cos_sin_for_prefill(self.max_position_embeddings)
        else:
            indices = batch_valid_length - 1
            freqs_cos, freqs_sin = \
                self.rotary_embedding.get_cos_sin_for_decode(indices,
                                                             self.max_position_embeddings)
        return self.rotary_embedding(query, key, freqs_cos, freqs_sin, batch_valid_length)


class OldLlama3RopeNet(nn.Cell):
    """A model class of old llama3 rotary embedding."""
    def __init__(self, config: TransformerConfig, rotary_cos_format=0, prefill: bool = True):
        super(OldLlama3RopeNet, self).__init__()
        self.seq_len = config.seq_length
        if config.seq_length > config.max_position_embeddings:
            self.max_position_embeddings = config.seq_length
        else:
            self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=self.seq_len,
                                  max_position_embedding=self.max_position_embeddings,
                                  theta=500000,
                                  rotary_dtype=mstype.float16,
                                  scaling_factor=dict(factor=8.0,
                                                      low_freq_factor=1.0,
                                                      high_freq_factor=4.0,
                                                      original_max_position_embeddings=128),
                                  extend_method=SeqExtendMethod.LLAMA3.value,
                                  is_dynamic=True)
        self.rotary_cos_format = rotary_cos_format
        self.prefill = prefill
        self.rotary_embedding = InferRotaryEmbedding(self.rotary_cos_format)

    def construct(self, query: Tensor, key: Tensor, batch_valid_length):
        if self.prefill:
            freqs_cis = self.freqs_mgr.prefill(query.shape[0], self.seq_len)
        else:
            freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        return self.rotary_embedding(query, key, freqs_cis, batch_valid_length)
