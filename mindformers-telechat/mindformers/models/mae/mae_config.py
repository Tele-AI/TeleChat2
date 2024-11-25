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
"""Mae Config API."""

from typing import Union

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['ViTMAEConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ViTMAEConfig(PretrainedConfig):
    """
    Config for Mae model

    Args:
        mask_ratio(float): The mask ratio of image, default 0.75.
        image_size(int): The size of image, default 224.
        patch_size(int): The patch size of image, default 16.
        num_channels(int): The channel number of image, default 3.
        initializer_range(float): The initializer range, default 0.02.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers(`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads(`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size(int): 3072,
        qkv_bias(bool): The QKV projection layer whether add bias, default True.
        hidden_act(str): The activation of the internal feedforward layer. Supports 'relu',
            'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
            'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
            If user wants to run the net in the parallel mode, the custom activation must also provide
            the `activation_shard` function. Please see the examples of the
            class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
        post_layernorm_residual(bool): Whether use post layernorm, defaylt False.
        layer_norm_eps(float): The epsilon value of the denominator. Default 1e-6.
        attention_probs_dropout_prob(float): The dropout ratio of attention layer, default 0.0.
        hidden_dropout_prob(float): The dropout ratio of hidden ffn layer, default 0.0.
        drop_path_rate(float): The dropout ratio of path, default 0.
        decoder_hidden_size(int): The hidden size of decoder layer, default 512.
        decoder_num_hidden_layers(int): The number of decoder hidden layers, default 8.
        decoder_num_attention_heads(`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_intermediate_size(int): 2048.
        norm_pix_loss(bool): True.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float32".
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.

    Returns:
        Class, ViTMAEConfig.
    """

    model_type = "mae"
    _support_list = MindFormerBook.get_config_support_list()['mae']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 mask_ratio: float = 0.75,
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
                 layer_norm_eps: float = 1e-6,
                 attention_probs_dropout_prob: float = 0.0,
                 hidden_dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.,
                 decoder_hidden_size: int = 512,
                 decoder_num_hidden_layers: int = 8,
                 decoder_num_attention_heads: int = 16,
                 decoder_intermediate_size: int = 2048,
                 norm_pix_loss: bool = True,
                 checkpoint_name_or_path: str = '',
                 layernorm_compute_type: mstype = mstype.float32,
                 softmax_compute_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float32,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 init_values=None,
                 window_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = num_channels
        self.initializer_range = initializer_range
        self.embed_dim = hidden_size
        self.depth = num_hidden_layers
        self.num_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout_rate = attention_probs_dropout_prob
        self.drop_rate = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.decoder_embed_dim = decoder_hidden_size
        self.decoder_depth = decoder_num_hidden_layers
        self.decoder_num_heads = decoder_num_attention_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pixel_loss = norm_pix_loss
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.init_values = init_values
        self.window_size = window_size
