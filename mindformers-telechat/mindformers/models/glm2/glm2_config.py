# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""ChatGLM2 config"""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype
from mindformers.mindformer_book import MindFormerBook

__all__ = ['ChatGLM2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ChatGLM2Config(PretrainedConfig):
    """
    ChatGLM2 model config class which defines the model size.

    Args:
        batch_size (int, optional): Batch size for input data, use in predict. Default: ``1``.
        num_layers (int, optional): Number of hidden layers in the Transformer encoder.
            Default: ``28``.
        padded_vocab_size (int, optional): Vocabulary size of the ChatGLM2 model. Default: ``65024``.
        hidden_size (int, optional): Dimensionality of the hidden layers. Default: ``4096``.
        ffn_hidden_size (int, optional): Dimensionality of the ffn layer. Default: ``13696``.
        kv_channels (int, optional): The number of channels for key and value vectors in the transformer.
            Default: ``128``.
        num_attention_heads (int, optional): The number of attention heads for each attention layer. Default: ``32``.
        seq_length (int, optional): The sequence length of input_ids, default is 2048. Default: ``2048``.
        hidden_dropout (float, optional): The dropout ratio of the dropout function. Default: ``0.0``.
        attention_dropout (float, optional): The dropout ratio for the attention matrix. Default: ``0.0``.
        layernorm_epsilon (float, optional): The ϵ value added to prevent the denominator from being zero when computing
            layer normalization. Default: ``1e-5``.
        rope_ratio (float, optional): RoPE rotation coefficient. Default: ``1``.
        rmsnorm (bool, optional): Whether to use rmsnorm. Default: ``True``.
        apply_residual_connection_post_layernorm (bool, optional): Whether apply the residual connection to post
            layernorm. Default: ``False``.
        post_layer_norm (bool, optional): Whether to use layer normalization after the ffn layer. Default: ``True``.
        add_bias_linear (bool, optional): Whether to add bias to the linear layer. Default: ``False``.
        add_qkv_bias (bool, optional): Whether to add bias for qkv. Default: ``True``.
        bias_dropout_fusion (bool, optional): Whether to add bias, dropout, and fusion operations. Default: ``True``.
        multi_query_attention (bool, optional): Whether to use multi query attention. Default: ``True``.
        multi_query_group_num (int, optional): Define multi group head attention heads number. Default: ``2``.
        apply_query_key_layer_scaling (bool, optional): Whether scaling the query_key layer. Default: ``True``.
        attention_softmax_in_fp32 (bool, optional): Whether apply fp32 to the attention softmax. Default: ``True``.
        fp32_residual_connection (bool, optional): Whether apply fp32 to residual connection layer. Default: ``False``.
        quantization_bit (int, optional): Weight and number of activation bits. Default: ``0``.
        pre_seq_len (int, optional): Length of the input sequence that can be learned. Default: ``None``.
        prefix_projection (bool, optional): Add a projection layer before a sequence. Default: ``False``.
        param_init_type (str, optional): Parameter initial dtype. Default: ``float16``.
        compute_dtype (str, optional): Linear layer compute dtype. Default: ``float16``.
        layernorm_compute_type (str, optional): LayerNorm compute dtype. Default: ``float32``.
        residual_dtype (str, optional): Residual compute dtype. Default: ``float32``.
        rotary_dtype (str, optional): Custom rotary position embedding compute dtype. Default: ``None``.
        use_past (bool, optional): Whether the model should use the past last key/values attentions (if applicable to
            the model) to speed up decoding. Default: ``False``.
        use_flash_attention (bool, optional): Whether enable flash attention ops, default False. Default: ``False``.
        enable_high_performance (bool, optional): Whether to adjust parallel strategies of qkv and ffn,
            in order to get high performance. Default: ``False``.
        block_size (int, optional): The maximum number of tokens in one block can have when using PagedAttention.
            Default: ``16``.
        num_blocks (int, optional): The maximum number of blocks when using PagedAttention. Default: ``128``.
        is_dynamic (bool, optional): Whether to use dynamic diagram mode. Default: ``False``.
        eos_token_id (int, optional): The token id of the *end-of-sequence* token. Default: ``2``.
        pad_token_id (int, optional): In multi-batch inference, the token id value used to pad shorter sequences to
            match the length of the longest sequence. Default: ``0``.
        gmask_token_id (int, optional): A special token representing a gmask token. Default: ``None``.
        bos_token_id (int, optional): The id of the *beginning-of-sequence* token. Default: ``None``.
        repetition_penalty (float, optional): The parameter for repetition penalty. 1.0 means no penalty.
            Default: ``1.0``.
        checkpoint_name_or_path (str, optional): Checkpoint path or name used to load to the network.
            Default: ``None``.
        parallel_config (TransformerOpParallelConfig, optional): The parallel configure. an instance of
            `TransformerOpParallelConfig` with default args. Default: ``TransformerOpParallelConfig``.
        offset (int, optional): The layer offset for each (mini) stage. Default: ``0``.
        pp_interleave_num (int, optional): Number of microbatch interleavings in pipeline parallelism. Default: ``1``.
        mlp_concat (bool, optional): Whether to concatenate two mlp to one Linear. Default: ``True``.
        qkv_concat (bool, optional): Whether to concatenate query key and value Linear calculation to one entire Linear.
             Default: ``True``.
        use_rearrange_rope (bool, optional): Whether to use rearranged rotary embedding. Default: ``False``.
        mask_generate (str, optional): Which mask generation to use, which can be "inmap", "compress_reset", None. When
             set as None, lower triangular mask is used. Default: ``None``.
        fine_grain_interleave (int, optional): Number of slices for fine grain interleave feature, which covers
             communication time with computation time in tensor parallel case. Default: ``1``.
        use_ring_attention (bool, optional): Whether enable ring attention ops. Default: ``False``.
        kwargs (dict, optional): A variable number of keyword parameters reserved for the keyword parameters to be
            expanded.

    Returns:
        An instance of ChatGLM2Config.

    Examples:
        >>> from mindformers.models import ChatGLM2Config
        >>> config = ChatGLM2Config(num_layers=2, seq_length=1024)
        >>> print(config.num_layers)
        2
        >>> print(config.seq_length)
        1024

    """


    model_type = "glm2"
    _support_list = MindFormerBook.get_config_support_list()['glm2']
    _support_list.extend(MindFormerBook.get_config_support_list()['glm3'])
    _support_list.extend(MindFormerBook.get_config_support_list()['codegeex2'])

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 batch_size=1,
                 num_layers=28,
                 padded_vocab_size=65024,
                 hidden_size=4096,
                 ffn_hidden_size=13696,
                 kv_channels=128,
                 num_attention_heads=32,
                 seq_length=2048,
                 hidden_dropout=0.0,
                 attention_dropout=0.0,
                 layernorm_epsilon=1e-5,
                 rope_ratio=1,
                 rmsnorm=True,
                 apply_residual_connection_post_layernorm=False,
                 post_layer_norm=True,
                 add_bias_linear=False,
                 add_qkv_bias=True,
                 bias_dropout_fusion=True,
                 multi_query_attention=True,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=True,
                 attention_softmax_in_fp32=True,
                 fp32_residual_connection=False,
                 quantization_bit=0,
                 pre_seq_len=None,
                 prefix_projection=False,
                 param_init_type: str = "float16",
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 residual_dtype: str = "float32",
                 rotary_dtype: str = None,
                 use_past=False,
                 use_flash_attention=False,
                 enable_high_performance=False,
                 block_size=16,
                 num_blocks=128,
                 is_dynamic=False,
                 eos_token_id=2,
                 pad_token_id=0,
                 gmask_token_id=None,
                 bos_token_id=None,
                 repetition_penalty=1.0,
                 checkpoint_name_or_path=None,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 offset: int = 0,
                 pp_interleave_num: int = 1,
                 mlp_concat: bool = True,
                 qkv_concat: bool = True,
                 use_rearrange_rope: bool = False,
                 mask_generate: str = None,
                 fine_grain_interleave: int = 1,
                 use_ring_attention: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rope_ratio = rope_ratio
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.residual_dtype = convert_mstype(residual_dtype)
        self.rotary_dtype = convert_mstype(rotary_dtype) if rotary_dtype is not None else self.compute_dtype
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.enable_high_performance = enable_high_performance
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty = repetition_penalty
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.gmask_token_id = gmask_token_id
        self.bos_token_id = bos_token_id
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.is_dynamic = is_dynamic
        self.num_heads = self.num_attention_heads
        self.n_kv_heads = self.multi_query_group_num if self.multi_query_attention else None
        self.offset = offset
        self.pp_interleave_num = pp_interleave_num
        self.mlp_concat = mlp_concat
        self.qkv_concat = qkv_concat
        self.use_rearrange_rope = use_rearrange_rope
        self.mask_generate = mask_generate
        self.fine_grain_interleave = fine_grain_interleave
        self.use_ring_attention = use_ring_attention
