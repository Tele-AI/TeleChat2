# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingsoftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Swin Config API."""

from typing import Union

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig, default_transformer_config, \
    default_moe_config
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['SwinConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class SwinConfig(PretrainedConfig):
    """
    Swin config class which defines the model size

    Args:
         image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
         patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
         num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
         embed_dim (`int`, *optional*, defaults to 128):
            Dimensionality of patch embedding.
         depths (`list(int)`, *optional*, defaults to [2, 2, 18, 2]):
            Depth of each layer in the Transformer encoder.
         num_heads (`list(int)`, *optional*, defaults to [4, 8, 16, 32]):
            Number of attention heads in each layer of the Transformer encoder.
         window_size (`int`, *optional*, defaults to 7):
            Size of windows.
         shift_size (`int`, *optional*, defaults to 0):
            The window shift size.
         mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
         qkv_bias (`bool`, *optional*, defaults to True):
            Whether a learnable bias should be added to the queries, keys and values.
         hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
         attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
         drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
         use_absolute_embeddings (`bool`, *optional*, defaults to False):
            Whether to add absolute position embeddings to the patch embeddings.
         patch_norm (`bool`, *optional*, defaults to `True`):
            Whether to use norm in SwinPatchEmbeddings.
         hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
         weight_init (`str`, *optional*, defaults to "normal"):
            Weight initialize type.
         num_labels (`int`, *optional*, defaults to 1000):
            The number of labels in downstream tasks.
         loss_type (`str`, *optional*, defaults to "SoftTargetCrossEntropy"):
            The type of loss function.
         param_init_type (`str`, *optional*, defaults to "float32"):
            Network parameter initialization type.
         moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.
         parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
         checkpoint_name_or_path (`string`, *optional*, defaults to "swin_base_p4w7):
            checkpoint path or name used to load to the network.

    Examples:
        >>> import os
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> from mindformers import SwinConfig
        >>> # init a config with a model name
        >>> config_a = SwinConfig.from_pretrained('swin_base_p4w7')
        >>> type(config_c)
        <class 'mindformers.models.swin.swin_config.SwinConfig'>
        >>> # init a config with a config path
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        ...                        'configs', 'swin', 'run_swin_base_p4w7_224_100ep.yaml')
        >>> config_b = SwinConfig.from_pretrained(config_path)
        >>> type(config_c)
        <class 'mindformers.models.swin.swin_config.SwinConfig'>
        >>> # init a config with args
        >>> config_c = SwinConfig()
        >>> type(config_c)
        <class 'mindformers.models.swin.swin_config.SwinConfig'>
    """

    model_type = "swin"
    _support_list = MindFormerBook.get_config_support_list()['swin']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 4,
                 num_channels: int = 3,
                 embed_dim: int = 128,
                 depths: list = (2, 2, 18, 2),
                 num_heads: list = (4, 8, 16, 32),
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 layer_norm_eps: float = 1e-5,
                 hidden_dropout_prob: float = 0.,
                 attention_probs_dropout_prob: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_absolute_embeddings: bool = False,
                 patch_norm: bool = True,
                 hidden_act: str = 'gelu',
                 weight_init: str = 'normal',
                 num_labels: int = 1000,
                 loss_type: str = "SoftTargetCrossEntropy",
                 param_init_type: mstype = mstype.float32,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: str = '',
                 **kwargs):
        """Swin Base Config"""
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_absolute_embeddings = use_absolute_embeddings
        self.patch_norm = patch_norm
        self.hidden_act = hidden_act
        self.weight_init = weight_init
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.param_init_type = param_init_type
        self.moe_config = moe_config
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        super(SwinConfig, self).__init__(**kwargs)
