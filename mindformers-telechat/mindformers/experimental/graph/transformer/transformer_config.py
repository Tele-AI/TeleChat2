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
"""Config Class"""
from typing import Callable

import mindspore.common.dtype as mstype

from mindformers.experimental.utils import init_method_normal


class ModelParallelConfig:
    r"""
        Parallel config class for setting parallel configuration, such as the data parallel and model parallel.

        Note:
            Except the recompute argument, other arguments will **not** be effective when the user doesn't set
            auto_parallel_context to `SEMI_AUTO_PARALLEL` or `AUTO_PARALLEL`.
            The micro_batch_num must be greater than or equal to pipeline_stage when training.
            The data_parallel\*model_parallel \*pipeline_stage must be equal or less equal to the device. When setting
            the pipeline stage and optimizer_shard, the config will overwrite the auto_parallel_context. When given the
            8 devices and the data_parallel is 1 and model_parallel is 1, the calculation will be repeated on each
            device.

        Args:
            data_parallel (int): The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int): The model parallel way. The parameters of dense layers in MultiheadAttention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            context_parallel (int): The context parallel way. The context data will be sliced into n parts for each
                layer according to the context parallel strategy. Default: 1.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. If True, the embedding lookup
                will be a data parallel style training and model_parallel value will be ignored.  If false, the
                embedding table will be sharded into n parts at the 0-th dimension row slice of the embedding table,
                where the n is the model parallel way determined by this parameter. Default: True

        Returns:
            Class, ModelParallelConfig.
    """

    config_name = "model_parallel_config"

    def __init__(self,
                 data_parallel: int = 1,
                 tensor_parallel: int = 1,
                 context_parallel: int = 1,
                 vocab_emb_dp: bool = True,
                 **kwargs):
        super(ModelParallelConfig, self).__init__(**kwargs)
        self.data_parallel = data_parallel
        self.tensor_parallel = tensor_parallel
        self.context_parallel = context_parallel
        self.vocab_emb_dp = vocab_emb_dp


class TransformerConfig(ModelParallelConfig):
    r"""
        Configuration object for mindformers transformers.

        The initialization function has an argument for each parameter, including those in ModelParallelConfig.

        Args:
            batch_size (int): batch size for input data. Default: 1.
            seq_length (int): The sequence length of input_ids. Default: 1.
            padded_vocab_size (int): Vocabulary size of the model. Default: 1.
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer. Default: 1.
            ffn_hidden_size (int): Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size if not
                provided.
            num_layers (int): Number of hidden layers in the Transformer encoder. Default: 1.
            num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
                Default: 1.
            num_query_groups (int): Define multi group head attention heads number. Default: None.
            max_position_embeddings (int): Sets the maximum position embedding size. Default: None.
            layernorm_epsilon (float): Epsilon value for any LayerNorm operations. Default: 1e-5.
            hidden_dropout (float): Dropout probability for transformer hidden state. Default: 0.0.
            attention_dropout (float): Post attention dropout probability. Default: 0.0.
            init_method_std (float): Standard deviation of the zero mean normal for the default initialization method,
                not used if init_method and output_layer_init_method are provided. Default: 0.01.
            rotary_percent (float): The proportion of the hidden dimension to which rotary positional embeddings will
                be applied.Default: 1.0.
            rotary_seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
                The value must be a float larger than 1.0. Default: None.
            compute_dtype (mstype): Linear layer compute dtype. Default: mstype.float32.
            softmax_compute_dtype (mstype): Softmax compute dtype. Default: mstype.float32.
            param_init_dtype (mstype): Parameter initial dtype. Default: mstype.float32.
            add_bias_linear (bool): Include a bias term in all linear layers (QKV projections, after core attention,
                and two in MLP layer). Default: False.
            add_qkv_bias (bool): Add a bias term only for QKV projections. Default: False.
            mlp_has_gate (bool): Apply gating in MLP block. Default: True.
            gated_linear_unit (bool): Use a gated linear unit for the first linear layer in the MLP. Default: False.
            fp16 (bool): If true, train with fp16 mixed precision training. Default: False.
            bf16 (bool): If true, train with bf16 mixed precision training. Default: False.
            group_query_attention (bool): Enable group query attention. Default: False.
            use_flash_attn (bool): Whether enable flash attention ops. Default: False.
            qkv_concat (bool): Whether to concatenate query, key, and value tensors. Default: False.
            use_attn_mask_compression (bool): Whether to use attention mask compression. Default: False.
            apply_query_key_layer_scaling (bool): If true, scale Q * K^T by 1 / layer-number. This improve numeric
                stability when training with fp16. Default: False.
            attention_softmax_in_fp32 (bool): If True, run attention masking and softmax in fp32. This should be True if
                apply_query_key_layer_scaling is True. Default: True.
            apply_residual_connection_post_layernorm (bool): If True, uses the original BERT residule connection
                ordering. Default: False.
            fp16_lm_cross_entropy (bool): If True, use FP16 precision for cross-entropy loss; otherwise, use FP32.
                Default: False.
            untie_embeddings_and_output_weights (bool): If True, the word embeddings and output layer weights are
                untied; otherwise, they share the same weights. Default: True.
            hidden_act (str): Specifies the activation function used in hidden layers. Default: "gelu".
            mask_func_type (str): Attention mask compute method. Default: "attn_mask_fill".
            normalization (str): Normalization used in transformerlayer block. Default: "RNSNorm".
            position_embedding_type (str): Defines the type of position embedding to use. Default: "rope".
            init_method (Callable): Method to initialize weights. Note that bias is always set to zero. Should be a
                function that takes a single Tensor and initializes it. Default: None.
            output_layer_init_method: Method to initialize weights of the output layer of both attention and MLP blocks.
                If None, will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
                init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers). Default: None.

        Returns:
            Class, TransformerConfig.
    """

    config_name = "transformer_config"

    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 1,
                 padded_vocab_size: int = 1,
                 hidden_size: int = 1,
                 ffn_hidden_size: int = None,
                 num_layers: int = 1,
                 num_attention_heads: int = 1,
                 num_query_groups: int = None,
                 max_position_embeddings: int = None,
                 layernorm_epsilon: float = 1e-5,
                 hidden_dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 init_method_std: float = 0.01,
                 rotary_percent: float = 1.0,
                 rotary_seq_len_interpolation_factor: float = None,
                 compute_dtype: mstype = mstype.float32,
                 softmax_compute_dtype: mstype = mstype.float32,
                 params_dtype: mstype = mstype.float32,
                 embedding_init_type: mstype = mstype.float32,
                 add_bias_linear: bool = False,
                 add_qkv_bias: bool = False,
                 mlp_has_gate: bool = True,
                 gated_linear_unit: bool = False,
                 fp16: bool = False,
                 bf16: bool = False,
                 group_query_attention: bool = False,
                 use_flash_attn: bool = False,
                 qkv_concat: bool = False,
                 use_attn_mask_compression: bool = False,
                 apply_query_key_layer_scaling: bool = False,
                 attention_softmax_in_fp32: bool = True,
                 apply_residual_connection_post_layernorm: bool = False,
                 fp16_lm_cross_entropy: bool = False,
                 untie_embeddings_and_output_weights: bool = True,
                 hidden_act: str = "gelu",
                 mask_func_type: str = "attn_mask_fill",
                 normalization: str = "FusedRMSNorm",
                 position_embedding_type: str = "rope",
                 init_method: Callable = None,
                 output_layer_init_method: Callable = None,
                 **kwargs):
        super(TransformerConfig, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.max_position_embeddings = max_position_embeddings
        self.layernorm_epsilon = layernorm_epsilon
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.init_method_std = init_method_std
        self.rotary_percent = rotary_percent
        self.rotary_seq_len_interpolation_factor = rotary_seq_len_interpolation_factor
        self.compute_dtype = compute_dtype
        self.softmax_compute_dtype = softmax_compute_dtype
        self.params_dtype = params_dtype
        self.embedding_init_type = embedding_init_type
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.mlp_has_gate = mlp_has_gate
        self.gated_linear_unit = gated_linear_unit
        self.fp16 = fp16
        self.bf16 = bf16
        self.group_query_attention = group_query_attention
        self.use_flash_attn = use_flash_attn
        self.qkv_concat = qkv_concat
        self.use_attn_mask_compression = use_attn_mask_compression
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = untie_embeddings_and_output_weights
        self.hidden_act = hidden_act
        self.mask_func_type = mask_func_type
        self.normalization = normalization
        self.position_embedding_type = position_embedding_type
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.post_init_checks()

    def post_init_checks(self):
        """Modify attributes after initialization."""
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.num_attention_heads % self.tensor_parallel != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_parallel})."
            )

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_parallel != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_parallel})."
            )

        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.seq_length

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std, self.params_dtype)

    def update(self):
        """Modify attributes after covert."""
        if self.ffn_hidden_size == 4:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.num_query_groups == 1:
            self.num_query_groups = self.num_attention_heads

        if self.max_position_embeddings == 1:
            self.max_position_embeddings = self.seq_length
