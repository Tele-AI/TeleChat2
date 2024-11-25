# Copyright 2022 Huawei Technologies Co., Ltd
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
CLIPConfig class, which consists of CLIPTextConfig and CLIPVisionConfig
All configs here are inherited from PretrainedConfig
"""
from typing import Optional

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.logger import logger


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class CLIPTextConfig(PretrainedConfig):
    r"""
    Config For CLIP Text Module

    Args:
        vocab_size (Optional[int]): Vocabulary size of the CLIP text model.
        hidden_size (Optional[int]): The dims of text features.
        intermediate_size (Optional[int]): Dimensionality of the "intermediate"
            (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (Optional[int]): The number of transformer layers in text encoder.
        num_attention_heads (Optional[int]): Number of attention heads for each
            attention layer in the Transformer encoder.
        max_position_embeddings (Optional[int]): The maximum sequence length that
            this model might ever be used with.
        hidden_act (Optional[str]): The non-linear activation function
            (function or string) in the encoder and pooler. Only "quick_gelu" supported currently.
        attention_dropout (Optional[float]): The dropout ratio for the attention probabilities.
        dropout (Optional[float]): The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        initializer_range (Optional[float]): The standard deviation of the
            truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (Optional[float]): A factor for initializing all weight matrices
            (should be kept to 1, used internally for initialization testing).

    Returns:
        Class, CLIPTextConfig
    """

    def __init__(self, vocab_size: Optional[int] = 49408,
                 hidden_size: Optional[int] = 512,
                 intermediate_size: Optional[int] = 2048,
                 num_hidden_layers: Optional[int] = 12,
                 num_attention_heads: Optional[int] = 8,
                 max_position_embeddings: Optional[int] = 77,
                 hidden_act: Optional[str] = "quick_gelu",
                 attention_dropout: Optional[float] = 0.0,
                 drop_out: Optional[float] = 0.0,
                 initializer_range: Optional[float] = 0.02,
                 initializer_factor: Optional[float] = 1.0,
                 **kwargs):
        super(CLIPTextConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.dropout = drop_out
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class CLIPVisionConfig(PretrainedConfig):
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
        Class, CLIPVisionConfig
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
                 **kwargs):
        super(CLIPVisionConfig, self).__init__(**kwargs)
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


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class CLIPConfig(PretrainedConfig):
    r"""
    Config For CLIP Model

    Args:
        text_config (Optional[CLIPTextConfig]): The config of text transformer.
        vision_config (Optional[CLIPVisionConfig]): The config of vision transformer.
        projection_dim (Optional[int]): The dims of projected featrues.
        logit_scale_init_value (Optional[float]): The initial value of the *logit_scale* parameter.
        checkpoint_name_or_path (Optional[str]): The path of checkpoint(.ckpt)
            or a support model name in CLIPConfig.show_support_list()
        dtype (Optional[str]): The type of tensors in model, ["float16", "float32"].

    Raises:
         TypeError: If the type of text_config is not CLIPTextConfig or the type of vision_config
            is not CLIPVisionConfig

    Returns:
        Class, CLIPConfig
    """

    model_type = "clip"
    _support_list = MindFormerBook.get_config_support_list()['clip']

    def __init__(self, text_config: Optional[CLIPTextConfig] = None,
                 vision_config: Optional[CLIPVisionConfig] = None, projection_dim: Optional[int] = 512,
                 logit_scale_init_value: Optional[float] = 2.6592,
                 checkpoint_name_or_path: Optional[str] = "", dtype: Optional[str] = "float16",
                 **kwargs):
        if text_config is None:
            text_config = CLIPTextConfig()
            logger.info("text_config is None. Initializing the CLIPTextConfig with default values.")
        elif isinstance(text_config, CLIPTextConfig):
            pass
        else:
            raise TypeError(f"text_config should be a "
                            f"CLIpTextConfig class, but got {type(CLIPTextConfig)}")

        if vision_config is None:
            vision_config = CLIPVisionConfig()
            logger.info("vision_config is None."
                        " Initializing the CLIPTextConfig with default values.")
        elif isinstance(vision_config, CLIPVisionConfig):
            pass
        else:
            raise TypeError("text_config should be a CLIPVisionConfig"
                            f" class, but got {type(CLIPVisionConfig)}")

        super(CLIPConfig, self).__init__(**kwargs)

        self.text_config = text_config
        self.vision_config = vision_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.dtype = dtype
