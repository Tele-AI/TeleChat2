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
        batch_size (int, optional): Batch size for input data, use in predict. Default: ``1``.
        seq_length (int, optional): The sequence length of input_ids. Default: ``2048``.
        hidden_size (int, optional): Dimensionality of the encoder layers and the pooler layer. Default: ``4096``.
        num_layers (int, optional): Number of hidden layers in the Transformer decoder. Default: ``32``.
        num_heads (int, optional): Number of attention heads for each attention layer in the Transformer decoder.
            Default: ``32``.
        n_kv_heads (int, optional): Define multi group head attention heads number. Default: ``None``.
        max_position_embedding (int, optional): Customize the maximum sequence length that the model can handle.
            Default: "None".
        intermediate_size (int, optional): Customize the number of dimension of the intermediate layer.
            Default: ``None``.
        vocab_size (int, optional): Vocabulary size of the llama model. Default: ``32000``.
        multiple_of (int, optional): Define SwiGLU hidden layer size multiples. Default: ``256``.
        ffn_dim_multiplier (int, optional): Define ffn layer dim multiples. Default: ``None``.
        rms_norm_eps (float, optional): The epsilon value of the denominator. Default: ``1e-5``.
        bos_token_id (int, optional): The id of the *beginning-of-sequence* token. Default: ``1``.
        eos_token_id (int, optional): The id of the *end-of-sequence* token. Default: ``2``.
        pad_token_id (int, optional): The id of the *padding* token. Default: ``0``.
        ignore_token_id (int, optional): The id of the *ignoring* token. Default: ``-100``.
        theta (float, optional): Frequency factors for sine and cosine functions in RoPE. Default: ``10000.0``.
        compute_dtype (str, optional): Linear layer compute dtype. Default: ``float16``.
        layernorm_compute_type (str, optional): Layernorm compute dtype. Default: ``float32``.
        softmax_compute_type (str, optional): Softmax compute dtype. Default: ``float32``.
        rotary_dtype (str, optional): RoPE compute dtype. Default: ``float32``.
        param_init_type (str, optional): Parameter initial dtype. Default: ``float16``.
        residual_dtype (str, optional): Residual compute dtype. Default: ``None``.
        embedding_init_type (str, optional): Embedding weight initial dtype. Default: ``None``.
        qkv_has_bias (bool, optional): Whether the Query, Key, and Value projection has bias. Default: ``False``.
        qkv_concat (bool, optional): Whether concatenate the Query, Key, and Value projection. Default: ``False``.
        attn_proj_has_bias (bool, optional): Whether the attn projection has bias. Default: ``False``.
        parallel_config (Union[dict, TransformerOpParallelConfig], optional): The parallel configuration.
            Default: ``default_transformer_config`` , an instance of `TransformerOpParallelConfig` with default args.
        moe_config (Union[dict, MoEConfig], optional): The MoE configuration. Default: ``default_moe_config`` ,
            an instance of `MoEConfig` with default args.
        use_past (bool, optional): Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding. Default: ``False``.
        extend_method (str, optional): The extent method of seq length in inference. Default: ``None``.
        scaling_factor (float, optional): Scaling factor to adjust the weights of the frequency factors in the sine
            and cosine functions. Default: ``1.0``.
        is_dynamic (bool, optional): Whether to use dynamic shape. Default: ``False``.
        use_rope_slice (bool, optional): Whether to enable RoPE slicing. Default: ``False``.
        use_flash_attention (bool, optional): Whether to enable flash attention ops. Default: ``False``.
        use_ring_attention (bool, optional): Whether to enable ring attention ops. Default: ``False``.
        use_attn_mask_compression (bool, optional): Whether to enable attention mask compression. Default: ``False``.
        use_eod_attn_mask_compression (bool, optional): Whether to enable eod attention mask compression.
            Default: ``False``.
        parallel_optimizer (bool, optional): Whether to enable optimizer parallism. Default: ``False``.
        fine_grain_interleave (int, optional): Set the number of fine-grained interleave. Default: ``1``.
        pp_interleave_num (int, optional): Set the number of pipeline interleave. Default: ``1``.
        offset (int, optional): Offset of transformer layer when set pipeline stage number. Default: ``0``.
        init_method_std (float, optional): The sigma value when using normal type to initialize Linear.
            Default: ``0.01``.
        checkpoint_name_or_path (str, optional): checkpoint path or name used to load to the network. Default: ``None``.
        repetition_penalty (float, optional): The parameter for repetition penalty. 1.0 means no penalty.
            See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`_ for more details. Default: ``1.0``.
        max_decode_length (int, optional): The maximum length the generated tokens can have. Corresponds to the
            length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            Default: ``1024``.
        block_size (int, optional): The maximum number of tokens in one block can have when using paged attention.
            Default: ``16``.
        num_blocks (int, optional): The maximum number of blocks when using paged attention. Default: ``512``.
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            Default: ``5``.
        top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation. Default: ``1.0``.
        do_sample (bool, optional): Whether to use sampling; use greedy decoding otherwise. Default: ``True``.
        quant_config (dict, optional): Quantitative configuration. Default: ``None``.
        tie_word_embeddings (bool, optional): Whether to tie input and output embeddings. Default: ``False``.
        llm_backend (str, optional): Llm boost backend. Default: ``None``.
        fused_rms_norm (bool, optional): Whether to use the RMSNorm of the fusion operator. Default: ``True``.
        input_sliced_sig (bool, optional): If input_ids and labels have been processed to equal to seq_length,
            input_sliced_sig should be True, if not, input_sliced_sig should be False. Default: ``False``.
        rmsnorm_compute_2d (bool, optional): Whether to use 2D Add in RMS_NORM. Default: ``False``.
        chunk_prefill (bool, optional): Whether to use prefill mixed decode inference. Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to calculate the loss of each token. Default: ``False``.
        pipeline_stage (dict, optional): A dict set the start_stage, stage_num, and offset of the model when
            pipeline parallelism. Default: ``None``.

    Returns:
        LlamaConfig, a LlamaConfig instance.

    Examples:
        >>> from mindformers.models import LlamaConfig
        >>> config = LlamaConfig(num_layers=2, seq_length=1024)
        >>> print(config.num_layers)
        2
        >>> print(config.seq_length)
        1024

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
                 residual_dtype: str = None,
                 embedding_init_type=None,
                 qkv_has_bias: bool = False,
                 qkv_concat: bool = False,
                 attn_proj_has_bias: bool = False,
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
                 use_eod_attn_mask_compression: bool = False,
                 parallel_optimizer: bool = False,
                 fine_grain_interleave: int = 1,
                 pp_interleave_num: int = 1,
                 offset: int = 0,
                 init_method_std: float = 0.01,
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
                 input_sliced_sig: bool = False,
                 rmsnorm_compute_2d: bool = False,
                 chunk_prefill: bool = False,
                 calculate_per_token_loss: bool = False,
                 pipeline_stage: dict = None,
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
        self.attn_proj_has_bias = attn_proj_has_bias
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        if residual_dtype is not None:
            self.residual_dtype = convert_mstype(residual_dtype)
        else:
            self.residual_dtype = self.compute_dtype
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
        self.use_eod_attn_mask_compression = use_eod_attn_mask_compression
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
        self.init_method_std = init_method_std
        self.input_sliced_sig = input_sliced_sig
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        self.chunk_prefill = chunk_prefill
        self.calculate_per_token_loss = calculate_per_token_loss
        if (pipeline_stage is not None and
                pipeline_stage["start_stage"] + pipeline_stage["stage_num"] <= parallel_config.pipeline_stage):
            self.start_stage = pipeline_stage["start_stage"]
            self.stage_num = pipeline_stage["stage_num"]
            self.offset = pipeline_stage["offset"]
        else:
            self.start_stage = 0
            self.stage_num = 0
