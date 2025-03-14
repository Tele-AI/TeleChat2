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
Internvl2 Config API
"""
import copy
from typing import Optional, Union

from mindspore._checkparam import args_type_check
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype

__all__ = ['InternVisionConfig', 'InternVLChatConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class InternVisionConfig(PretrainedConfig):
    r"""
    InternViT config class which defines the vision model size.

    Args:
        image_size (Optional[int]): Image size input for the vision transformer, default is 448.
        hidden_size (Optional[int]): Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (Optional[int]): Number of hidden layers in the transformer encoder.
        num_attention_heads (Optional[int]): Number of attention heads for each attention layer.
        patch_size (Optional[int]): Size of each patch of the input image, default is 14.
        layer_norm_eps (Optional[float]): The epsilon value of layer normalization.
        dropout (Optional[float]): Dropout rate, default is 0.0.
        attention_dropout (Optional[float]): Dropout rate for attention layers, default is 0.0.
        hidden_act (Optional[str]): Activation function to use, default is "gelu".
        qk_normalization (Optional[bool]): Whether to use QK normalization, default is True.
        use_flash_attn (Optional[bool]): Whether to use flash attention ops, default is True.
        torch_dtype (Optional[str]): Data type for PyTorch, default is "bfloat16".

    Returns:
        InternViTConfig class.
    """

    model_type = 'intern_vit_6b'

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 image_size: Optional[int] = 448,
                 hidden_size: Optional[int] = 3200,
                 num_hidden_layers: Optional[int] = 45,
                 num_channels: Optional[int] = 3,
                 num_attention_heads: Optional[int] = 25,
                 patch_size: Optional[int] = 14,
                 layer_norm_eps: Optional[float] = 1e-06,
                 dropout: Optional[float] = 0.0,
                 drop_path_rate: Optional[float] = 0.0,
                 attention_dropout: Optional[float] = 0.0,
                 initializer_factor: Optional[float] = 0.1,
                 initializer_range: Optional[float] = 1e-10,
                 intermediate_size: Optional[int] = 12800,
                 norm_type: Optional[str] = "rms_norm",
                 hidden_act: Optional[str] = "gelu",
                 qk_normalization: bool = True,
                 qkv_bias: bool = False,
                 compute_dtype: Optional[str] = "float32",
                 param_init_type: Optional[str] = "float32",
                 use_flash_attn: bool = True,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: Optional[str] = None,
                 **kwargs):
        super(InternVisionConfig, self).__init__(**kwargs)

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.qk_normalization = qk_normalization
        self.qkv_bias = qkv_bias
        self.compute_dtype = convert_mstype(compute_dtype)
        self.param_init_type = convert_mstype(param_init_type)
        self.use_flash_attn = use_flash_attn
        self.parallel_config = parallel_config
        self.drop_path_rate = drop_path_rate
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.norm_type = norm_type
        self.num_channels = num_channels
        self.checkpoint_name_or_path = checkpoint_name_or_path


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class InternVLChatConfig(PretrainedConfig):
    r"""
    Config For InternVL Module

    Args:
        model_type (Optional[int]):
            model type for llava model, default is 'llava'.
        batch_size (Optional[int]):
            batch size for input data, use in predict.
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
        Class, InternVLChatConfig.
    """
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            vision_model=None,
            text_model=None,
            seq_length: int = 8192,
            batch_size: int = 1,
            compute_dtype: str = "bfloat16",
            layernorm_compute_type: str = "float32",
            softmax_compute_type: str = "float32",
            rotary_dtype: str = "bfloat16",
            param_init_type: str = "bfloat16",
            use_past: bool = False,
            use_backbone_lora: int = 0,
            use_llm_lora: int = 0,
            select_layer: int = -1,
            force_image_size: int = None,
            downsample_ratio: int = 0.5,
            template: str = None,
            dynamic_image_size: bool = False,
            use_thumbnail: bool = False,
            qkv_concat: bool = False,
            checkpoint_name_or_path: str = None,
            is_dynamic: bool = False,
            bos_token_id: int = 1,
            eos_token_id: int = 7,
            pad_token_id: int = 0,
            repetition_penalty: float = 1.0,
            block_size: int = 16,
            num_blocks: int = 512,
            top_k: int = 5,
            top_p: float = 1.0,
            do_sample: bool = False,
            max_decode_length: int = 512,
            ignore_token_id: int = -100,
            num_queries: int = 576,
            parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
            **kwargs):
        super(InternVLChatConfig, self).__init__(**kwargs)

        self.vision_model = vision_model
        self.text_model = text_model
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.qkv_concat = qkv_concat
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.param_init_type = convert_mstype(param_init_type)
        self.parallel_config = parallel_config
        self.use_past = use_past
        self.is_dynamic = is_dynamic
        self.repetition_penalty = repetition_penalty
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self.do_sample = do_sample
        self.num_queries = num_queries
        self.ignore_token_id = ignore_token_id
        self.max_decode_length = max_decode_length
        self.seq_length = seq_length
        self.vocab_size = text_model.vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.img_context_token_id = text_model.img_context_token_id

        # pass configs to submodule config
        self.text_model.model_config.use_past = use_past
        self.text_model.model_config.compute_dtype = "bfloat16"
        self.text_model.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.text_model.model_config.softmax_compute_type = self.softmax_compute_type
        self.text_model.model_config.rotary_dtype = self.rotary_dtype
        self.text_model.model_config.param_init_type = self.param_init_type
        self.text_model.model_config.seq_length = seq_length
        self.text_model.model_config.batch_size = batch_size
        self.text_model.model_config.ignore_token_id = ignore_token_id
        self.text_model.model_config.parallel_config = parallel_config
        self.text_model.model_config.is_dynamic = is_dynamic
        self.text_model.model_config.block_size = block_size
        self.text_model.model_config.num_blocks = num_blocks
        self.text_model.model_config.repetition_penalty = self.repetition_penalty
        self.text_model.model_config.top_k = self.top_k
        self.text_model.model_config.top_p = self.top_p
        self.text_model.model_config.do_sample = self.do_sample
        self.text_model.model_config.max_decode_length = self.max_decode_length
        self.text_model.model_config.bos_token_id = self.bos_token_id
        self.text_model.model_config.eos_token_id = self.eos_token_id
        self.text_model.model_config.pad_token_id = self.pad_token_id

        self.vision_model.model_config.use_past = False
        self.vision_model.model_config.parallel_config = parallel_config
        self.vision_model.model_config.compute_dtype = self.compute_dtype
        self.vision_model.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.vision_model.model_config.softmax_compute_type = self.softmax_compute_type
        self.vision_model.model_config.rotary_dtype = self.rotary_dtype
        self.vision_model.model_config.param_init_type = self.param_init_type
        self.vision_model.model_config.is_dynamic = is_dynamic
        self.vision_model.model_config.batch_size = batch_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_model'] = self.vision_model.to_dict()
        output['text_model'] = self.text_model.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail

        return output
