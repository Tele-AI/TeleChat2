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
# This file was refer to project:
# https://github.com/facebookresearch/mae
# ============================================================================
"""ViT Model."""
import math
import numpy as np
from mindspore import load_param_into_net, Parameter, nn
from mindspore import ops as P
from mindspore import dtype as mstype
import mindspore.common.initializer as weight_init
from mindformers.mindformer_book import MindFormerBook
from mindformers.core.loss import build_loss
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.vit.vit_modules import Block, LayerNorm, Linear, Dropout, PixelShuffle
from mindformers.models.vit.vit_modules import PatchEmbed
from mindformers.models.vit.vit_config import ViTConfig


class VitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTModel(VitPreTrainedModel):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    The supported model name could be selected from ViTConfig.show_support_list().

    Args:
        config (ViTConfig): the config of Vit model.

    Examples:
        >>> from mindformers import ViTModel, AutoConfig
        >>> # input model name, load model and weights
        >>> model_a = ViTModel.from_pretrained('vit_base_p16')
        >>> type(model_a)
        <class 'mindformers.models.vit.vit.ViTModel'>
        >>> # input config, load model without weights
        >>> config = AutoConfig.from_pretrained('vit_base_p16')
        >>> model_b = ViTModel(config)
        >>> type(model_b)
        <class 'mindformers.models.vit.vit.ViTModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['vit']

    def __init__(self, config=None):
        config = config if config else ViTConfig()
        super().__init__(config)
        self.use_moe = (config.moe_config.expert_num > 1)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        self.global_pool = config.use_mean_pooling
        self.patch_embed = PatchEmbed(img_size=config.image_size, patch_size=config.patch_size,
                                      in_features=config.num_channels, out_features=config.hidden_size,
                                      parallel_config=parallel_config)
        self.cls_tokens = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=config.initializer_range),
                                    (1, 1, config.hidden_size)), requires_grad=True)
        num_patches = self.patch_embed.num_patches
        seq_length = num_patches + 1
        self.seq_length = seq_length
        self.num_patches = num_patches
        self.num_masked = num_patches - seq_length + 1
        self.pos_embed = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=config.initializer_range),
                                    (1, seq_length, config.hidden_size)), requires_grad=True)
        # stochastic depth decay rule
        hdr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        self.blocks = nn.CellList([
            Block(hidden_size=config.hidden_size,
                  ffn_hidden_size=config.intermediate_size,
                  seq_length=seq_length,
                  drop_rate=config.hidden_dropout_prob,
                  attention_dropout_rate=config.attention_probs_dropout_prob,
                  hidden_dropout_rate=hdr[i],
                  layer_norm_eps=config.layer_norm_eps,
                  qkv_bias=config.qkv_bias,
                  init_values=config.init_values,
                  weight_init='XavierUniform',
                  layernorm_compute_type=config.layernorm_compute_type,
                  softmax_compute_type=config.softmax_compute_type,
                  window_size=None,
                  num_heads=config.num_attention_heads,
                  hidden_act=config.hidden_act,
                  post_layernorm_residual=config.post_layernorm_residual,
                  param_init_type=config.param_init_type,
                  parallel_config=parallel_config_args)
            for i in range(config.num_hidden_layers)])

        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.cast = P.Cast()
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.cat = P.Concat(axis=1)
        self.fc_norm = LayerNorm((config.hidden_size,), eps=1e-6).shard(((dp, 1),))

        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1),))
        self.dropout = Dropout(keep_prob=(1. - config.hidden_dropout_prob))
        self.dropout.shard(((dp, 1, 1),))

        self.stride_slice = P.StridedSlice().shard(((dp, 1, 1),))

        self.init_weights_vit()
        self.fix_init_weight()

    def fix_init_weight(self):
        """fix init weight"""

        def rescale(param, layer_id):
            values = param.data / (math.sqrt(2.0 * layer_id))
            param.set_data(values)

        for layer_id, block in enumerate(self.blocks):
            if self.use_moe:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.ffn.projection.weight, layer_id + 1)
            else:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.projection.weight, layer_id + 1)

    def init_weights_vit(self):
        """init weights vit. ViT weight initialization, original timm impl (for reproducibility) """
        for name, cell in self.cells_and_names():

            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init.initializer(
                    weight_init.TruncatedNormal(sigma=self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            if name == "patch_embed.proj":
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_tokens'}

    def load_pretrained(self, params_dict):
        return load_param_into_net(self, params_dict)

    def construct_without_pool(self, image, mask=None):
        """construct of vit without pool"""
        tokens = self.patch_embed(image, mask)
        batch_size = image.shape[0]
        cls_tokens = self.tile(self.cls_tokens, (batch_size, 1, 1))
        tokens = self.cat((cls_tokens, tokens))
        if self.pos_embed is not None:
            tokens = self.add(tokens, self.pos_embed)

        x = self.dropout(tokens)
        encoder_input_mask = P.Ones()((batch_size, self.seq_length, self.seq_length), mstype.float32)
        for block in self.blocks:
            x = block(x, encoder_input_mask)
        return x

    def construct(self, image):
        """construct of vit"""
        x = self.construct_without_pool(image)
        b, s, c = x.shape

        if self.global_pool:
            x = self.stride_slice(
                x, (0, 1, 0), (b, s, c), (1, 1, 1)
            )
            x = self.reduce_mean(x, 1)
            out = self.fc_norm(x)
        else:
            out = self.stride_slice(
                x, (0, 0, 0), (b, 1, c), (1, 1, 1)
            )
        return out


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTForImageClassification(VitPreTrainedModel):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    The supported model name could be selected from ViTConfig.show_support_list().

    Args:
        config (ViTConfig): the config of Vit model.

    Examples:
        >>> # input model name, load model and weights
        >>> model_a = ViTForImageClassification.from_pretrained('vit_base_p16')
        >>> # input config, load model without weights
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('vit_base_p16')
        >>> model_b = ViTForImageClassification(config)
    """

    _support_list = MindFormerBook.get_model_support_list()['vit']

    def __init__(self, config=None):
        config = config if config else ViTConfig()
        super().__init__(config)
        self.vit = ViTModel(config)
        self.head = Linear(
            config.hidden_size, config.num_classes,
            weight_init=weight_init.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)
        self.loss = build_loss(class_name=config.loss_type)
        self.load_checkpoint(config)

    def construct(self, image, target=None):
        """construct of vit"""
        out = self.vit(image)
        out = self.head(out)
        if not self.training:
            return out, target
        loss = self.loss(out, target)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTForMaskedImageModeling(VitPreTrainedModel):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    The supported model name could be selected from ViTConfig.show_support_list().

    Args:
        config (ViTConfig): the config of Vit model.

    Examples:
        >>> # input model name, load model and weights
        >>> model_a = ViTForMaskedImageModeling.from_pretrained('vit_base_p16')
        >>> # input config, load model without weights
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('vit_base_p16')
        >>> model_b = ViTForMaskedImageModeling(config)
    """
    _support_list = MindFormerBook.get_model_support_list()['vit']

    def __init__(self, config=None):
        config = config if config else ViTConfig()
        super().__init__(config)
        self.vit = ViTModel(config)
        self.vit.patch_embed = PatchEmbed(img_size=config.image_size, patch_size=config.patch_size,
                                          in_features=config.num_channels, out_features=config.hidden_size,
                                          use_mask=True, parallel_config=config.parallel_config)
        self.decoder = nn.CellList(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride ** 2 * config.num_channels,
                kernel_size=1,
            ),
            PixelShuffle(config.encoder_stride),
        )

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.l1_loss = nn.L1Loss(reduction='none')
        # Initialize weights and apply final processing
        self.init_weights_vit()

    def init_weights_vit(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    def construct(self, image, mask=None):
        """construct of vit for MIM"""
        x = self.vit.construct_without_pool(image)
        b, s, c = x.shape
        height = width = math.floor(s ** 0.5)
        x = self.reshape(self.transpose(x, (0, 2, 1)), (b, c, height, width))
        reconstruct_images = self.decoder(x)
        if not self.training:
            return reconstruct_images
        size = self.config.image_size // self.config.patch_size
        mask = self.reshape(mask, (-1, size, size))
        mask = P.repeat_elements(mask, self.config.patch_size, 1)
        mask = P.repeat_elements(mask, self.config.patch_size, 2)
        mask = self.expand_dims(mask, 1)
        reconstruction_loss = self.l1_loss(image, reconstruct_images)
        masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.in_chans
        return masked_im_loss
