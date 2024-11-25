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
# This file was refer to project:
# https://github.com/facebookresearch/mae
# ============================================================================
"""EVA-02 models' APIs."""
import numpy as np

from mindspore import Parameter, nn, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.modules.layers import LayerNorm

from .eva_config import EVA02Config
from .eva_module import PatchEmbed, RotaryEmbeddingCat, EvaBlock

__all__ = ['EVAModel']


class EVA02PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EVA02Config
    base_model_prefix = "eva02"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class EVAModel(EVA02PreTrainedModel):
    """
    A Transformer-based visual representation pre-trained model to reconstruct strong and robust
    language-aligned vision features via masked image modeling.

    Args:
        config (EVA02Config): the config of EVA model.
    """

    def __init__(self, config=None):
        config = config if config else EVA02Config()
        super().__init__(config)

        self.post_norm = config.post_norm
        self.with_cls_token = config.with_cls_token

        # build model
        self.num_prefix_tokens = 1 if config.class_token else 0
        self.patch_embed = PatchEmbed(image_size=config.image_size,
                                      patch_size=config.patch_size,
                                      in_features=config.num_channels,
                                      out_features=config.hidden_size,
                                      compute_dtype=config.compute_dtype)
        num_patches = self.patch_embed.num_patches

        self.cls_token = None
        if config.class_token:
            self.cls_token = Parameter(
                Tensor(np.zeros(shape=(1, 1, config.hidden_size)), config.param_init_type),
                requires_grad=True, name='cls_token'
            )

        self.pos_embed = None
        if config.use_abs_pos_emb:
            pos_shape = (1, num_patches + self.num_prefix_tokens, config.hidden_size)
            self.pos_embed = Parameter(
                Tensor(np.zeros(shape=pos_shape), config.param_init_type),
                requires_grad=True, name='pos_embed'
            )

        self.rope = None
        if config.use_rot_pos_emb:
            ref_feat_shape = (config.ref_feat_shape, config.ref_feat_shape)
            self.rope = RotaryEmbeddingCat(head_dim=config.hidden_size // config.num_attention_heads,
                                           feat_shape=self.patch_embed.grid_size,
                                           ref_feat_shape=ref_feat_shape,
                                           in_pixels=False,
                                           rotary_emb_type=config.rotary_emb_type)

        # stochastic depth decay rule
        dpr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        if config.layer_norm == 'rmsnorm':
            layer_norm = LlamaRMSNorm
        else:
            layer_norm = LayerNorm

        self.blocks = nn.CellList([
            EvaBlock(hidden_size=config.hidden_size,
                     num_heads=config.num_attention_heads,
                     mlp_hidden_size=config.intermediate_size,
                     qkv_bias=config.qkv_bias,
                     swiglu_mlp=config.use_swiglu,
                     scale_mlp=config.use_scale_mlp,
                     proj_drop=config.hidden_dropout_prob,
                     attn_drop=config.attention_dropout_prob,
                     drop_path=dpr[i],
                     layer_norm=layer_norm,
                     layer_norm_eps=config.layer_norm_eps,
                     use_qkv_fused=config.use_qkv_fused,
                     use_qkv_simple=config.use_qkv_simple,
                     use_attn_norm=config.use_attn_norm,
                     use_post_norm=config.use_post_norm,
                     compute_dtype=config.compute_dtype,
                     layer_norm_type=config.layer_norm_type,
                     param_init_type=config.param_init_type)
            for i in range(config.num_hidden_layers)])

        if self.post_norm:
            self.norm = nn.Identity()
        else:
            self.norm = LlamaRMSNorm((config.hidden_size,))

        self.tile = P.Tile()
        self.cat = P.Concat(axis=1)
        self.add = P.Add()
        self.cast = P.Cast()
        self.stride_slice = P.StridedSlice()
        self.reduce_mean = P.ReduceMean()

    def _pos_embed(self, x):
        """Apply positional embedding on patches."""
        b, _, _ = F.shape(x)
        pos_embed = self.pos_embed
        if self.rope is not None:
            rot_pos_embed = self.rope.get_pos_embed()
        else:
            rot_pos_embed = None

        if self.cls_token is not None:
            cls_tokens = self.tile(self.cls_token, (b, 1, 1))
            x = self.cat((self.cast(cls_tokens, F.dtype(x)), x))
        if pos_embed is not None:
            x = self.add(x, pos_embed)
        # leak module to obtain shared rotary position embedding and apply patch dropout
        return x, rot_pos_embed

    def construct_features(self, image):
        """Get image feature by forwarding EVAModel."""
        x = self.patch_embed(image)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def construct(self, image):
        """Forward of EVAModel."""
        x = self.construct_features(image)
        if not self.with_cls_token:
            b, l, c = F.shape(x)
            x = self.stride_slice(x, (0, 1, 0), (b, l, c), (1, 1, 1))
        return x
