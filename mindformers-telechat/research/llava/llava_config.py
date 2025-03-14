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
"""Llava Config API"""
from typing import Optional, Union

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.llama import LlamaConfig
from mindformers.models.utils import convert_mstype
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer.transformer import default_transformer_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['LlavaConfig', "LlavaCLIPConfig"]


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlavaCLIPConfig(PretrainedConfig):
    r"""
    Config For CLIP Vision Module

    Args:
        hidden_size (Optional[int]): Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (Optional[int]): Dimensionality of the "intermediate"
            (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (Optional[int]): Number of hidden layers in the Transformer encoder.
        num_attention_heads (Optional[int]): Number of attention heads for
            each attention layer in the Transformer encoder.
        image_size (Optional[int]): The size (resolution) of each image.
        patch_size (Optional[int]): The size (resolution) of each patch.
        hidden_act (Optional[str]): The non-linear activation function
            (function or string) in the encoder and pooler. Only "quick_gelu" supported currently.
        dropout (Optional[float]): The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_dropout (Optional[float]): The dropout ratio for the attention probabilities.
        initializer_range (Optional[float]): The standard deviation of the
            truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (Optional[float]): A factor for initializing all weight matrices
            (should be kept to 1, used internally for initialization testing).

    Returns:
        Class, LlavaCLIPConfig
    """

    def __init__(self, hidden_size: Optional[int] = 768,
                 intermediate_size: Optional[int] = 3072,
                 num_hidden_layers: Optional[int] = 12,
                 num_attention_heads: Optional[int] = 12,
                 image_size: Optional[int] = 224,
                 patch_size: Optional[int] = 32,
                 hidden_act: Optional[str] = "quick_gelu",
                 dropout: Optional[float] = 0.0,
                 attention_dropout: Optional[float] = 0.0,
                 initializer_range: Optional[float] = 0.02,
                 initializer_factor: Optional[float] = 1.0,
                 vision_feature_layer: Optional[int] = -2,
                 vision_feature_select_strategy: Optional[str] = "default",
                 checkpoint_name_or_path: str = "",
                 **kwargs):
        super(LlavaCLIPConfig, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlavaConfig(PretrainedConfig):
    r"""
    Config For llava Module

    Args:
        model_type (Optional[int]):
            model type for llava model, default is 'llava'.
        batch_size (Optional[int]):
            batch size for input data, use in predict.
        freeze_vision (Optional[bool]):
            whether to freeze vit weights, default is True.
        freeze_llm (Optional[bool]):
            whether to freeze LLM weights, default is True.
        freeze_resampler (Optional[bool]):
            whether to freeze Adapter weights, default is True.
        prompt (Optional[str]):
            prompt for llama model.
        prompt_length (Optional[int]):
            prompt length for llama model.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        rotary_dtype (Optional[str]):
            rope compute dtype, default is "float16".
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float16".
        vision_config (Optional[ViTConfig]):
            config for ViTModel.
        text_config (Optional[LlamaConfig]):
            config for LLM model, like llama.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        is_training (Optional[bool]): whether the model is in training state.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        is_dynamic (`bool`, *optional*, defaults to `False`):
            Whether the model use dynamic inputs or not.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        block_size (`int`, *optional*, defaults to 16):
            The maximum number of tokens in one block can have when using paged attention.
        num_blocks (`int`, *optional*, defaults to 512):
            The maximum number of blocks when using paged attention.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        ignore_token_id (Optional[int]): The id of the *ignoring* token.
        num_queries (Optional[int]): The image seq length
    Returns:
        Class, Blip2Config.
    """
    def __init__(self,
                 model_type: str = "llava",
                 batch_size: int = 8,
                 seq_length: int = 4096,
                 freeze_vision: bool = True,
                 freeze_llm: bool = True,
                 freeze_resampler: bool = True,
                 prompt: bool = False,
                 prompt_length: int = 0,
                 checkpoint_name_or_path: str = None,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float16",
                 param_init_type: str = "float16",
                 vision_config: Optional[LlavaCLIPConfig] = None,
                 text_config: Optional[LlamaConfig] = None,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 is_training: bool = True,
                 use_past: bool = False,
                 is_dynamic: bool = False,
                 repetition_penalty: float = 1.0,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = False,
                 max_decode_length: int = 512,
                 ignore_token_id: int = -100,
                 num_queries: int = 576,
                 **kwargs):
        super(LlavaConfig, self).__init__(**kwargs)

        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)

        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.param_init_type = convert_mstype(param_init_type)
        self.model_type = model_type
        self.batch_size = batch_size
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.freeze_resampler = freeze_resampler
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.prompt = prompt
        self.prompt_length = prompt_length

        self.parallel_config = parallel_config
        self.is_training = is_training
        self.use_past = use_past
        self.is_dynamic = is_dynamic
        self.repetition_penalty = repetition_penalty
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.ignore_token_id = ignore_token_id
        self.max_decode_length = max_decode_length
        self.seq_length = seq_length

        self.text_config = text_config
        self.vision_config = vision_config

        # pass configs to submodule config
        self.text_config.model_config.compute_dtype = self.compute_dtype
        self.text_config.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.text_config.model_config.softmax_compute_type = self.softmax_compute_type
        self.text_config.model_config.rotary_dtype = self.rotary_dtype
        self.text_config.model_config.compute_dtype = self.compute_dtype
        self.text_config.model_config.param_init_type = self.param_init_type
        self.text_config.model_config.seq_length = seq_length
        self.text_config.model_config.batch_size = batch_size
        self.text_config.model_config.ignore_token_id = ignore_token_id
        self.text_config.model_config.parallel_config = parallel_config
        self.text_config.model_config.use_past = self.use_past
        self.text_config.model_config.is_dynamic = self.is_dynamic
        self.text_config.model_config.block_size = self.block_size
        self.text_config.model_config.num_blocks = self.num_blocks
        self.text_config.model_config.repetition_penalty = self.repetition_penalty
        self.text_config.model_config.top_k = self.top_k
        self.text_config.model_config.top_p = self.top_p
        self.text_config.model_config.do_sample = self.do_sample
        self.text_config.model_config.top_p = self.top_p
        self.text_config.model_config.max_decode_length = self.max_decode_length
        self.vocab_size = text_config.model_config.vocab_size

        self.vision_config.model_config.parallel_config = parallel_config
        self.vision_config.model_config.compute_dtype = self.compute_dtype
        self.vision_config.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.vision_config.model_config.softmax_compute_type = self.softmax_compute_type
        self.vision_config.model_config.rotary_dtype = self.rotary_dtype
        self.vision_config.model_config.param_init_type = self.param_init_type
        self.vision_config.model_config.is_dynamic = is_dynamic

        self.pad_token_id = text_config.model_config.pad_token_id
        self.eos_token_id = text_config.model_config.eos_token_id
        self.num_queries = num_queries
