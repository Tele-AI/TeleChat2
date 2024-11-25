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
"""Llama Config API."""

from typing import Optional, Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, \
    TransformerOpParallelConfig, default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype
from mindformers.mindformer_book import MindFormerBook

__all__ = ['LlamaConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlamaConfig(PretrainedConfig):
    """
    Llama config class which defines the model size.

    Args:
        batch_size (int, optional): batch size for input data, use in predict. Default: ``1`` .
        seq_length (int, optional): The sequence length of input_ids. Default: ``2048`` .
        vocab_size (int, optional): Default: ``32000`` .
            Vocabulary size of the BERT model.
        hidden_size (int, optional):
            Dimensionality of the encoder layers and the pooler layer. Default: ``4096`` .
        num_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Default: ``32`` .
        num_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder. Default: ``32`` .
        multiple_of (int, optional): Define SwiGLU hidden layer size multiples. Default: ``256`` .
        n_kv_heads (int, optional): Define multi group head attention heads number. Default: ``None`` .
        ffn_dim_multiplier (int, optional): Define ffn layer dim multiples. Default: ``None`` .
        rms_norm_eps (float, optional): The epsilon value of the denominator. Default: ``1e-5`` .
        bos_token_id (int, optional): The id of the *beginning-of-sequence* token. Default: ``1`` .
        eos_token_id (int, optional): The id of the *end-of-sequence* token. Default: ``2`` .
        pad_token_id (int, optional): The id of the *padding* token. Default: ``0`` .
        ignore_token_id (int, optional): The id of the *ignoring* token. Default: ``-100`` .
        compute_dtype (str, optional):
            Linear layer compute dtype. Default: ``float16`` .
        layernorm_compute_type (str, optional):
            layernorm compute dtype. Default: ``float32`` .
        softmax_compute_type (str, optional):
            softmax compute dtype. Default: ``float32`` .
        rotary_dtype (str, optional):
            rope compute dtype. Default: ``float32`` .
        param_init_type (str, optional):
            parameter initial dtype. Default: ``float16`` .
        qkv_has_bias (bool, optional):
            Whether the Query, Key, and Value projection has bias. Default: ``False`` .
        use_past (bool, optional):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding. Default: ``False`` .
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default: ``default_transformer_config`` ,
            an instance of `TransformerOpParallelConfig` with default args.
        extend_method(str, optional): The extent method of seq length of inference. Default: ``None`` .
        use_flash_attention(bool, optional): Whether enable flash attention ops. Default: ``False`` .
        use_ring_attention(bool, optional): Whether enable ring attention ops. Default: ``False`` .
        offset(int, optional): Offset of transformer layer when set pipeline stage number. Default: ``0`` .
        checkpoint_name_or_path (str, optional):
            checkpoint path or name used to load to the network. Default: ``None`` .
        repetition_penalty (float, optional):
            The parameter for repetition penalty. 1.0 means no penalty.
            See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`_ for more details. Default: ``1.0`` .
        max_decode_length (int, optional):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set. Default: ``1024`` .
        top_k (int, optional):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Default: ``5`` .
        top_p (float, optional):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation. Default: ``1.0`` .
        do_sample (bool, optional):
            Whether to use sampling; use greedy decoding otherwise. Default: ``True`` .
        block_size (int, optional):
            The maximum number of tokens in one block can have when using paged attention. Default: ``16`` .
        num_blocks (int, optional):
            The maximum number of blocks when using paged attention. Default: ``512`` .
        tie_word_embeddings (bool, optional):
            Whether to tie input and output embeddings. Default: ``False`` .
        llm_backend (str, optional):
            Llm boost backend. Default: ``None`` .
        fused_rms_norm (bool, optional):
            Whether or not to use the RMS_NORM of the fusion operator. Default: ``True`` .

    Returns:
        LlamaConfig, a LlamaConfig instance.

    Examples:
        >>> from mindformers.models import LlamaConfig
        >>> config = LlamaConfig(num_layers=2, seq_length=1024)
        >>> print(config)
        LlamaConfig {
            "batch_size": 1,
            "block_size": 16,
            "bos_token_id": 1,
            "checkpoint_name_or_path": "",
            "compute_dtype": "float16",
            "do_sample": true,
            "embedding_init_type": "float16",
            "eos_token_id": 2,
            "extend_method": "None",
            "ffn_dim_multiplier": null,
            "fine_grain_interleave": 1,
            "hidden_size": 4096,
            "ignore_token_id": -100,
            "intermediate_size": null,
            "is_dynamic": false,
            "layernorm_compute_type": "float32",
            "llm_backend": "",
            "max_decode_length": 1024,
            "max_position_embedding": 1024,
            "mindformers_version": "dev",
            "model_type": "llama",
            "multiple_of": 256,
            "n_kv_heads": null,
            "num_blocks": 512,
            "num_heads": 32,
            "num_layers": 2,
            "offset": 0,
            "pad_token_id": 0,
            "parallel_decoding_params": null,
            "parallel_optimizer": false,
            "param_init_type": "float16",
            "pp_interleave_num": 1,
            "qkv_concat": false,
            "qkv_has_bias": false,
            "quant_config": null,
            "repetition_penalty": 1.0,
            "rms_norm_eps": 1e-05,
            "rotary_dtype": "float32",
            "scaling_factor": 1.0,
            "seq_length": 1024,
            "softmax_compute_type": "float32",
            "theta": 10000.0,
            "tie_word_embeddings": false,
            "top_k": 5,
            "top_p": 1.0,
            "use_attn_mask_compression": false,
            "use_flash_attention": false,
            "use_past": false,
            "use_ring_attention": false,
            "use_rope_slice": false,
            "vocab_size": 32000
            }

    """

    model_type = "llama"
    _support_list = MindFormerBook.get_config_support_list()['llama']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 2048,
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 n_kv_heads: Optional[int] = None,
                 max_position_embedding: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 vocab_size: int = 32000,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 rms_norm_eps: float = 1e-5,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 0,
                 ignore_token_id: int = -100,
                 theta: float = 10000.0,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float32",
                 param_init_type: str = "float16",
                 embedding_init_type=None,
                 qkv_has_bias: bool = False,
                 qkv_concat: bool = False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_past: bool = False,
                 extend_method: str = "None",
                 scaling_factor: float = 1.0,
                 is_dynamic: bool = False,
                 use_rope_slice: bool = False,
                 use_flash_attention: bool = False,
                 use_ring_attention: bool = False,
                 use_attn_mask_compression: bool = False,
                 parallel_optimizer: bool = False,
                 fine_grain_interleave: int = 1,
                 pp_interleave_num: int = 1,
                 offset: int = 0,
                 checkpoint_name_or_path: str = "",
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 quant_config: dict = None,
                 tie_word_embeddings: bool = False,
                 llm_backend: str = "",
                 fused_rms_norm: bool = True,
                 **kwargs):
        """
        Note:
            vocab_size: int = 32000,  # defined later by tokenizer
            multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        """
        super(LlamaConfig, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embedding = max_position_embedding if max_position_embedding else seq_length
        self.intermediate_size = intermediate_size
        self.multiple_of = multiple_of
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rms_norm_eps = rms_norm_eps
        self.qkv_concat = qkv_concat
        self.param_init_type = convert_mstype(param_init_type)
        if embedding_init_type is not None:
            self.embedding_init_type = convert_mstype(embedding_init_type)
        else:
            self.embedding_init_type = self.param_init_type
        self.qkv_has_bias = qkv_has_bias
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.use_past = use_past
        self.extend_method = extend_method
        self.scaling_factor = scaling_factor
        self.is_dynamic = is_dynamic
        self.use_rope_slice = use_rope_slice
        self.use_flash_attention = use_flash_attention
        self.use_ring_attention = use_ring_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.parallel_optimizer = parallel_optimizer
        self.fine_grain_interleave = fine_grain_interleave
        self.pp_interleave_num = pp_interleave_num
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.theta = theta
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.quant_config = quant_config
        self.tie_word_embeddings = tie_word_embeddings
        self.llm_backend = llm_backend
        self.parallel_decoding_params = kwargs.get('parallel_decoding_params')
        self.fused_rms_norm = fused_rms_norm
