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
"""InternLM2 Config API."""
from typing import Optional, Union, List

from mindformers import LlamaConfig, TransformerOpParallelConfig
from mindformers.models.utils import convert_mstype
from mindformers.modules.transformer.transformer import default_transformer_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class InternLM2Config(LlamaConfig):
    """InternLM2 config class.

    Args:
        batch_size (Optional[int]): batch size for input data, use in predict.
        seq_length (Optional[int]): The sequence length of input_ids, default is 1024.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the BERT model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        multiple_of (Optional[int]): Define SwiGLU hidden layer size multiples, default 256.
        n_kv_heads (Optional[int]): Define multi group head attention heads number, default None.
        ffn_dim_multiplier (Optional[int]): Define ffn layer dim multiples, default None.
        rms_norm_eps (Optional[float]): The epsilon value of the denominator. Default 1e-5.
        bos_token_id (Optional[int]): The id of the *beginning-of-sequence* token.
        eos_token_id (Optional[int]): The id of the *end-of-sequence* token.
        pad_token_id (Optional[int]): The id of the *padding* token.
        ignore_token_id (Optional[int]): The id of the *ignoring* token.
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        rotary_dtype (Optional[str]):
            rope compute dtype, default is "float32".
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float16".
        has_bias (Optional[bool]):
            Whether the Query, Key, Value, Out projection has bias.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        extend_method(str): The extend method of seq length of inferencem,default None.
        use_flash_attention(bool): Whether enable flash attention ops, default False.
        fine_grain_interleave(int): The number of fine grain interleave, default 1.
        block_size(int): The shape of blocks in paged attention, default 16.
        num_blocks(int): The total number of blocks in paged attention, default 512.
        offset(int): Offset of transformer layer when set pipeline stage number.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling; use greedy decoding otherwise.

        Returns:
            Class, LlamaConfig.
    """

    model_type = "InternLM2ForCausalLM"

    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 2048,
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 n_kv_heads: Optional[int] = 8,
                 max_position_embedding: Optional[int] = None,
                 intermediate_size: Optional[int] = 14336,
                 vocab_size: int = 92544,  # defined later by tokenizer
                 multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
                 ffn_dim_multiplier: Optional[int] = None,
                 rms_norm_eps: float = 1e-5,
                 bos_token_id: int = 1,
                 eos_token_id: Union[int, List[int]] = 2,
                 pad_token_id: int = 2,
                 ignore_token_id: int = -100,
                 theta: float = 1000000.0,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float32",
                 param_init_type: str = "float16",
                 qkv_concat: bool = True,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 use_past: bool = False,
                 extend_method: str = "None",
                 scaling_factor: float = 1.0,
                 is_dynamic: bool = False,
                 use_rope_slice: bool = False,
                 use_flash_attention: bool = False,
                 fine_grain_interleave: int = 1,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 offset: int = 0,
                 checkpoint_name_or_path: str = "",
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
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
        self.param_init_type = convert_mstype(param_init_type)
        self.qkv_concat = qkv_concat
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
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
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.theta = theta
        self.block_size = block_size
        self.num_blocks = num_blocks
