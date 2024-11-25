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
"""Mae Config API."""

from typing import Union

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.models.utils import convert_mstype
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ViTConfig(PretrainedConfig):
    """
    Config for ViT model

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        hidden_act (`str` or `Callable`, *optional*, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        post_layernorm_residual (`bool`, *optional*, defaults to `False`):
            Whether to use post layernorm in Transformer.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout rate applied to the attention probs.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout rate of the dropout function on the bias dropout.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            drop path rate of transformer blocks
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to use mean pooling.
        num_labels (`int`, *optional*, defaults to 1000):
            The number of labels in downstream tasks.
        loss_type (`str`, *optional*, defaults to "SoftTargetCrossEntropy"):
            The type of loss function.
        encoder_stride (`int`, *optional*, defaults to 16):
            Factors that increase spatial resolution in the decoder header for mask image modeling
        checkpoint_name_or_path (`str`, *optional*, defaults to ""):
            checkpoint path or name used to load to the network.
        layernorm_compute_type (`str`, *optional*, defaults to "float32"):
            layernorm compute dtype.
        softmax_compute_type (`str`, *optional*, defaults to "float32"):
            softmax compute dtype.
        param_init_type (`float`, *optional*, defaults to "float32"):
            The type of parameters initializer.
        parallel_config (TransformerOpParallelConfig, defaults to default_transformer_config):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        moe_config (MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.

    Examples:
        >>> import os
        >>> from mindformers import ViTConfig
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> # init a config with a model name
        >>> config_a = ViTConfig.from_pretrained('vit_base_p16')
        >>> type(config_a)
        <class 'mindformers.models.vit.vit_config.ViTConfig'>
        >>> # init a config with a config path
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        ...                        'configs', 'vit', 'run_vit_base_p16_224_100ep.yaml')
        >>> config_b = ViTConfig.from_pretrained(config_path)
        >>> type(config_b)
        <class 'mindformers.models.vit.vit_config.ViTConfig'>
        >>> # init a config with args
        >>> config_c = ViTConfig()
        >>> type(config_c)
        <class 'mindformers.models.vit.vit_config.ViTConfig'>
    """

    model_type = "vit"
    _support_list = MindFormerBook.get_config_support_list()['vit']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 initializer_range: float = 0.02,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 qkv_bias: bool = True,
                 hidden_act: str = "gelu",
                 post_layernorm_residual: bool = False,
                 layer_norm_eps: float = 1e-12,
                 attention_probs_dropout_prob: float = 0.0,
                 hidden_dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.1,
                 use_mean_pooling: bool = True,
                 num_classes: int = 1000,
                 loss_type: str = "SoftTargetCrossEntropy",
                 encoder_stride: int = 16,
                 checkpoint_name_or_path: str = '',
                 layernorm_compute_type: mstype = mstype.float32,
                 softmax_compute_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float32,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 init_values=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.layer_norm_eps = layer_norm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.use_mean_pooling = use_mean_pooling
        self.loss_type = loss_type
        self.encoder_stride = encoder_stride
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.param_init_type = convert_mstype(param_init_type)
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.init_values = init_values
