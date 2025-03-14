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
"""DeepSeekV3 Config API."""
from typing import Optional, Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, \
    TransformerOpParallelConfig, default_moe_config
from mindformers.models.utils import convert_mstype
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook
from deepseek2_config import DeepseekV2Config

__all__ = ['DeepseekV3Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class DeepseekV3Config(DeepseekV2Config):
    """DeepSeekV3 config class which defines the model size. It redefines the attention block and experts in moe blocks.
    NOTE: If any nuisances, refer to in-between llama_config and the transformer original deepseek configuration.

    Args:
        batch_size (int, optional): Batch size for input data, use in predict. Default: ``1``.
        seq_length (int, optional): The sequence length of input_ids. Default: ``2048``.
        hidden_size (int, optional): Dimensionality of the encoder layers and the pooler layer. Default: ``4096``.
        num_layers (int, optional): Number of hidden layers in the Transformer decoder. Default: ``32``.
        num_heads (int, optional): Number of attention heads for each attention layer in the Transformer decoder.
            Default: ``32``.
        n_kv_heads (int, optional): Define multi group head attention heads number. Default: ``None``.
        max_position_embedding (int, optional): Customize the maximum sequence length that the model can handle.
            Default: ``32768``.
        intermediate_size (int, optional): Customize the number of dimension of the intermediate layer.
            Default: ``None``.
        multiple_of (Optional[int]): Define SwiGLU hidden layer size multiples, default 256.
        ffn_dim_multiplier (Optional[int]): Define ffn layer dim multiples, default None.
        kv_lora_rank (int, optional): kv_lora_rank for Multi-Latent-Attention. Default: ``512``.
        q_lora_rank (int, optional): q_lora_rank for Multi-Latent-Attention. Default: ``1536``.
        qk_rope_head_dim (int, optional): qk_rope_head_dim for Multi-Latent-Attention. Default: ``64``.
        v_head_dim (int, optional): v_head_dim for Multi-Latent-Attention. Default: ``128``.
        qk_nope_head_dim (int, optional): qk_nope_head_dim for Multi-Latent-Attention. Default: ``128``.
        vocab_size (int, optional): Vocabulary size of the llama model. Default: ``32000``.
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
        qkv_has_bias (bool, optional): Whether the Query, Key, and Value projection has bias. Default: ``False``.
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
        use_flash_attention (bool, optional): Whether to enable flash attention ops. Default: ``False``.
        fine_grain_interleave (int, optional): Set the number of fine-grained interleave. Default: ``1``.
        pp_interleave_num (int, optional): Set the number of pipeline interleave. Default: ``1``.
        offset (int, optional): Offset of transformer layer when set pipeline stage number. Default: ``0``.
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
        mtp_depth (`int`): The depth for multi-token prediction.
        mtp_loss_factor (`float`): The loss factor for multi-token prediction.
        return_extra_loss (`bool`): Whether to use extra loss for moe modules.

        Returns:
            Class, DeepseekV3Config.
    """
    model_type = "deepseekv3"
    _support_list = MindFormerBook.get_config_support_list()['deepseekv3']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 2048,
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 n_kv_heads: Optional[int] = None,
                 max_position_embeddings: int = 32768,
                 intermediate_size: int = 18432,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 kv_lora_rank: int = 512,
                 q_lora_rank: int = 1536,
                 qk_rope_head_dim: int = 64,
                 v_head_dim: int = 128,
                 qk_nope_head_dim: int = 128,
                 vocab_size: int = 129280,
                 rms_norm_eps: float = 1e-6,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 0,
                 ignore_token_id: int = -100,
                 theta: float = 10000.0,
                 compute_dtype: str = "bfloat16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float32",
                 param_init_type: str = "bfloat16",
                 init_method_std=0.006,
                 qkv_has_bias=False,
                 qkv_concat=False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_past: bool = False,
                 extend_method: str = "YARN",
                 scaling_factor: dict = None,
                 is_dynamic: bool = False,
                 use_flash_attention: bool = False,
                 fine_grain_interleave: int = 1,
                 pp_interleave_num: int = 1,
                 offset: int = 0,
                 checkpoint_name_or_path="",
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 mtp_depth: int = 0,
                 mtp_loss_factor: float = 0.3,
                 return_extra_loss: bool = True,
                 input_sliced_sig: bool = False,
                 use_fused_rope=False,
                 use_fused_swiglu=False,
                 enable_fa_var_len=False,
                 **kwargs):
        super(DeepseekV3Config, self).__init__(**kwargs)
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
        self.max_position_embeddings = max_position_embeddings if max_position_embeddings else seq_length
        self.intermediate_size = intermediate_size

        # new features of mla attention
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

        self.n_kv_heads = n_kv_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rms_norm_eps = rms_norm_eps
        self.qkv_concat = qkv_concat
        self.param_init_type = convert_mstype(param_init_type)
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
        self.use_flash_attention = use_flash_attention
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
        self.return_extra_loss = return_extra_loss
        self.mtp_depth = mtp_depth
        self.mtp_loss_factor = mtp_loss_factor
        self.input_sliced_sig = input_sliced_sig
        self.mlp_has_gate = True
        self.hidden_act = "silu"
        self.init_method_std = init_method_std
        self.use_fused_rope = use_fused_rope
        self.use_fused_swiglu = use_fused_swiglu
        self.enable_fa_var_len = enable_fa_var_len
