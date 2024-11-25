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
"""EVA-02 Config API."""
from typing import Union

from mindspore import dtype as mstype

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig, \
    default_transformer_config, default_moe_config
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype

__all__ = ['EVA02Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class EVA02Config(PretrainedConfig):
    """
    EVA-02 model config class.
    """

    model_type = "eva02"

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 2048,
                 ref_feat_shape: int = 16,
                 num_classes: int = 512,
                 hidden_dropout_prob: float = 0.0,
                 attention_dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.,
                 drop_rate: float = 0.0,
                 layer_norm: str = 'layernorm',
                 layer_norm_eps: float = 1e-6,
                 class_token: bool = True,
                 use_abs_pos_emb: bool = True,
                 use_rot_pos_emb: bool = True,
                 qkv_bias: bool = True,
                 use_swiglu: bool = True,
                 use_scale_mlp: bool = True,
                 use_qkv_fused: bool = False,
                 use_qkv_simple: bool = False,
                 use_attn_norm: bool = True,
                 use_post_norm: bool = False,
                 post_norm: bool = False,
                 with_cls_token: bool = False,
                 checkpoint_name_or_path: str = '',
                 compute_dtype: mstype = mstype.float16,
                 layer_norm_type: mstype = mstype.float32,
                 rotary_emb_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float16,
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
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.ref_feat_shape = ref_feat_shape

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm
        self.layer_norm_eps = float(layer_norm_eps)

        self.class_token = class_token
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rot_pos_emb = use_rot_pos_emb

        self.qkv_bias = qkv_bias
        self.use_post_norm = use_post_norm
        self.use_swiglu = use_swiglu
        self.use_scale_mlp = use_scale_mlp
        self.use_qkv_fused = use_qkv_fused
        self.use_qkv_simple = use_qkv_simple
        self.use_attn_norm = use_attn_norm

        self.post_norm = post_norm
        self.with_cls_token = with_cls_token
        self.num_classes = num_classes
        self.checkpoint_name_or_path = checkpoint_name_or_path

        self.compute_dtype = convert_mstype(compute_dtype)
        self.layer_norm_type = convert_mstype(layer_norm_type)
        self.rotary_emb_type = convert_mstype(rotary_emb_type)
        self.param_init_type = convert_mstype(param_init_type)

        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.init_values = init_values
