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
"""Qformer Config API."""


import mindspore.common.dtype as mstype

from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.bert import BertConfig


__all__ = ['QFormerConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QFormerConfig(BertConfig):
    """
    Qformer config class which defines the model size
    """

    model_type = "blip_2_qformer"

    def __init__(self,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 query_length: int = 32,
                 resize_token_embeddings: bool = True,
                 special_token_nums: int = 1,
                 vocab_size: int = 30523,
                 hidden_size: int = 768,
                 encoder_width: int = 1408,
                 head_embed_dim: int = 256,
                 bos_token_id: int = 30522,
                 sep_token_id: int = 102,
                 pad_token_id: int = 0,
                 max_position_embeddings: int = 512,
                 layer_norm_eps: float = 1.e-12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 chunk_size_feed_forward: int = 0,
                 cross_attention_freq: int = 2,
                 intermediate_size: int = 3072,
                 initializer_range: float = 0.02,
                 hidden_act: str = "gelu",
                 dtype: str = "float32",
                 layernorm_dtype: str = "float32",
                 softmax_dtype: str = "float32",
                 compute_dtype: str = "float16",
                 add_cross_attention: bool = True,
                 use_relative_positions: bool = False,
                 tie_word_embeddings: bool = True,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 convert_param_from_bert: bool = False,
                 parallel_config: str = "default",
                 moe_config: str = "default",
                 **kwargs):
        super(QFormerConfig, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.query_length = query_length
        self.resize_token_embeddings = resize_token_embeddings
        self.special_token_nums = special_token_nums
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_width = encoder_width
        self.head_embed_dim = head_embed_dim
        self.bos_token_id = bos_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.cross_attention_freq = cross_attention_freq
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.dtype = mstype.float32 if dtype == "float32" else mstype.float16
        self.layernorm_dtype = mstype.float32 if layernorm_dtype == "float32" else mstype.float16
        self.softmax_dtype = mstype.float32 if softmax_dtype == "float32" else mstype.float16
        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.add_cross_attention = add_cross_attention
        self.use_relative_positions = use_relative_positions
        self.tie_word_embeddings = tie_word_embeddings
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.convert_param_from_bert = convert_param_from_bert

        self.parallel_config = default_transformer_config if parallel_config == "default" \
                                                          else parallel_config
        self.moe_config = default_moe_config if moe_config == "default" else moe_config

        # additional args, not commonly used.
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.position_embedding_type = "absolute" if not use_relative_positions else "relative"
        self.loss_reduction = kwargs.pop("loss_reduction", "mean")
