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
"""Llama Vision Config API."""

from typing import Optional, Union, List
from mindformers.modules.transformer.transformer import default_transformer_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.utils import convert_mstype
from mindformers.mindformer_book import MindFormerBook

__all__ = ['MllamaConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class MllamaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a MllamaVisionModel. It is used to instantiate an
    Mllama vision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mllama-11B.

    Args:
        hidden_size (int, optional): Dimensionality of the encoder layers and the pooler layer. Default: ``1280``.
        hidden_act (str or function, optional): The non-linear activation function (function or string)
            in the encoder and pooler. If string, "gelu", "relu", "selu" and "gelu_new",
            "quick_gelu" are supported. defaults to "gelu".
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Default: ``32``.
        num_global_layers (int, optional): Number of global layers in the Transformer encoder.
            Vision model has a second transformer encoder, called global. Default: ``8``.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer. Default: ``16``.
        num_channels (int, optional): Number of channels in the input image. Default: ``3``.
        intermediate_size (int, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer. Default: ``5120``.
        vision_output_dim (int, optional): Dimensionality of the vision model output. Includes output of
            transformer encoder with intermediate layers and global transformer encoder. Default: ``7680``.
        image_size (int, optional): The size (resolution) of each image tile. Default: ``560``.
        patch_size (int, optional): The size (resolution) of each patch. Default: ``14``.
        norm_eps (float, optional): The epsilon used by the layer normalization layers. Default: ``1e-5``.
        max_num_tiles (int, optional): Maximum number of tiles for image splitting. Default: ``4``.
        intermediate_layers_indices (List[int], optional): Indices of intermediate layers of transformer
            encoder from which to extract and output features. These output features are concatenated
            with final hidden state of transformer encoder. Default: [3, 7, 15,23, 30]
        supported_aspect_ratios (List[List[int]], optional): List of supported aspect ratios for
            image splitting. If not specified, the default supported aspect ratios are
            [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]] for max_num_tiles=4.
        initializer_range (float, optional): The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices. Default: ``0.02``.

    """

    model_type = "mllama_vision"

    def __init__(self,
                 hidden_size: Optional[int] = 1280,
                 hidden_act: Optional[str] = "gelu",
                 num_hidden_layers: Optional[int] = 32,
                 num_global_layers: Optional[int] = 8,
                 num_attention_heads: Optional[int] = 16,
                 num_channels: Optional[int] = 3,
                 intermediate_size: Optional[int] = 5120,
                 vision_output_dim: Optional[int] = 7680,
                 image_size: Optional[int] = 560,
                 patch_size: Optional[int] = 14,
                 norm_eps: Optional[float] = 1e-5,
                 max_num_tiles: Optional[int] = 4,
                 max_num_images: Optional[int] = 1,
                 output_attentions: bool = False,
                 intermediate_layers_indices: Optional[List[int]] = None,
                 supported_aspect_ratios: Optional[List[List[int]]] = None,
                 initializer_range: Optional[float] = 0.02,
                 **kwargs):
        super(MllamaVisionConfig, self).__init__(**kwargs)
        if supported_aspect_ratios is None:
            if max_num_tiles != 4:
                raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")
            supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if intermediate_layers_indices is None:
            intermediate_layers_indices = [3, 7, 15, 23, 30]

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.max_num_images = max_num_images
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.intermediate_layers_indices = intermediate_layers_indices
        self.num_global_layers = num_global_layers
        self.max_num_tiles = max_num_tiles
        self.norm_eps = norm_eps
        self.attention_heads = num_attention_heads
        self.supported_aspect_ratios = supported_aspect_ratios
        self.initializer_range = initializer_range
        self.max_aspect_ratio_id = len(self.supported_aspect_ratios)
        self.output_attentions = output_attentions


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class MllamaTextConfig(LlamaConfig):
    """
    Mllama text config class which defines the model size.

    Args:
        cross_attention_layers (`List[int]`, *optional*):
            Indices of the cross attention layers. If not specified, will default to[3, 8, 13, 18, 23, 28, 33, 38].

    Returns:
        MllamaTextConfig, a MllamaTextConfig instance.

    """

    model_type = "mllama_text"

    def __init__(self, cross_attention_layers: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.cross_attention_layers = cross_attention_layers


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class MllamaConfig(PretrainedConfig):
    r"""
    Mllama config class which defines the model size.

    """
    model_type = "mllama"
    _support_list = MindFormerBook.get_config_support_list()['mllama']

    def __init__(self,
                 model_type: str = "mllama",
                 batch_size: int = 8,
                 seq_length: int = 4096,
                 freeze_vision: bool = True,
                 checkpoint_name_or_path: str = None,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float16",
                 param_init_type: str = "float16",
                 vision_model: Optional[MllamaVisionConfig] = None,
                 text_model: Optional[MllamaTextConfig] = None,
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
                 use_flash_attention: bool = False,
                 max_decode_length: int = 512,
                 ignore_token_id: int = -100,
                 **kwargs):
        super(MllamaConfig, self).__init__(**kwargs)
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
        self.checkpoint_name_or_path = checkpoint_name_or_path

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
        self.use_flash_attention = use_flash_attention

        self.vision_model = vision_model
        self.text_model = text_model

        # self.vision_model.model_config.parallel_config = parallel_config
        self.vision_model.model_config.compute_dtype = self.compute_dtype
        self.vision_model.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.vision_model.model_config.softmax_compute_type = self.softmax_compute_type
        self.vision_model.model_config.rotary_dtype = self.rotary_dtype
        self.vision_model.model_config.param_init_type = self.param_init_type
        self.vision_model.model_config.parallel_config = parallel_config
        self.vision_model.model_config.use_flash_attention = self.use_flash_attention

        self.text_model.model_config.compute_dtype = self.compute_dtype
        self.text_model.model_config.layernorm_compute_type = self.layernorm_compute_type
        self.text_model.model_config.softmax_compute_type = self.softmax_compute_type
        self.text_model.model_config.rotary_dtype = self.rotary_dtype
        self.text_model.model_config.param_init_type = self.param_init_type
        self.text_model.model_config.seq_length = seq_length
        self.text_model.model_config.batch_size = batch_size
        self.text_model.model_config.ignore_token_id = ignore_token_id
        self.text_model.model_config.parallel_config = parallel_config
        self.text_model.model_config.use_past = self.use_past
        self.text_model.model_config.is_dynamic = self.is_dynamic
        self.text_model.model_config.block_size = self.block_size
        self.text_model.model_config.num_blocks = self.num_blocks
        self.text_model.model_config.repetition_penalty = self.repetition_penalty
        self.text_model.model_config.top_k = self.top_k
        self.text_model.model_config.top_p = self.top_p
        self.text_model.model_config.do_sample = self.do_sample
        self.text_model.model_config.top_p = self.top_p
        self.text_model.model_config.max_decode_length = self.max_decode_length
        self.text_model.model_config.use_flash_attention = self.use_flash_attention
        self.vocab_size = text_model.model_config.vocab_size
