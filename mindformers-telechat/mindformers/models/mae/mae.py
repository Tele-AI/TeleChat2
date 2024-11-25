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
# This file was refer to project:
# https://github.com/facebookresearch/mae
# ============================================================================
"""Mae Model."""
import numpy as np
from mindspore import Tensor, Parameter, nn
from mindspore import dtype as mstype
from mindspore import ops as P
import mindspore.common.initializer as weight_init
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.mae.mae_modules import Block, LayerNorm, Linear
from mindformers.models.mae.mae_modules import PatchEmbed, Patchify, UnPatchify
from mindformers.models.mae.mae_modules import get_2d_sincos_pos_embed
from mindformers.core.loss import MSELoss
from mindformers.models.mae.mae_config import ViTMAEConfig


__all__ = ['ViTMAEModel', 'ViTMAEForPreTraining']


class MAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTMAEConfig
    base_model_prefix = "mae"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTMAEModel(MAEPreTrainedModel):
    """
    Pretrain MAE Module.
    The supported model name could be selected from ViTMAEConfig.show_support_list().

    Args:
        config (ViTMAEConfig): the config of Mae model.

    Examples:
        >>> # input model name
        >>> model_a = ViTMAEModel.from_pretrained('mae_vit_base_p16')
        >>> # input config
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('mae_vit_base_p16')
        >>> model_b = ViTMAEModel(config)
    """
    _support_list = MindFormerBook.get_model_support_list()['mae']

    def __init__(self, config=None):
        config = config if config else ViTMAEConfig()
        super().__init__(config)
        self.use_moe = (config.moe_config.expert_num > 1)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size=config.image_size, patch_size=config.patch_size,
                                      in_features=config.in_chans, out_features=config.embed_dim,
                                      parallel_config=config.parallel_config)
        num_patches = self.patch_embed.num_patches
        seq_length = int((1 - config.mask_ratio) * num_patches) + 1
        self.seq_length = seq_length
        self.num_masked = num_patches - seq_length + 1
        self.cls_tokens = Parameter(
            weight_init.initializer(weight_init.Normal(sigma=config.initializer_range), (1, 1, config.embed_dim)),
            requires_grad=True)
        self.num_patches = num_patches
        self.pos_embed = Parameter(
            weight_init.initializer(weight_init.Zero(), (1, num_patches + 1, config.embed_dim)),
            name='pos_embed', requires_grad=True)
        # stochastic depth decay rule
        hdr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.depth)]
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        self.blocks = nn.CellList([
            Block(hidden_size=config.embed_dim,
                  ffn_hidden_size=config.intermediate_size,
                  seq_length=seq_length,
                  drop_rate=config.drop_rate,
                  attention_dropout_rate=config.attention_dropout_rate,
                  hidden_dropout_rate=hdr[i],
                  layer_norm_eps=config.layer_norm_eps,
                  qkv_bias=config.qkv_bias,
                  init_values=config.init_values,
                  weight_init='XavierUniform',
                  layernorm_compute_type=config.layernorm_compute_type,
                  softmax_compute_type=config.softmax_compute_type,
                  window_size=config.window_size,
                  num_heads=config.num_heads,
                  hidden_act=config.hidden_act,
                  moe_config=config.moe_config,
                  post_layernorm_residual=config.post_layernorm_residual,
                  param_init_type=config.param_init_type,
                  parallel_config=parallel_config_args)
            for i in range(config.depth)])
        self.norm = LayerNorm((config.embed_dim,), eps=config.layer_norm_eps).shard(((dp, 1, 1),))
        # --------------------------------------------------------------------------

        self.stride_slice = P.StridedSlice().shard(((1, 1, 1),))
        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.gather = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))

        self.init_weights()
        self.init_pos_emd()

    def init_pos_emd(self):
        """init values of pos_embed"""
        encoder_pos_emd = Tensor(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        encoder_pos_emd = P.ExpandDims()(encoder_pos_emd, 0)
        self.pos_embed = Parameter(encoder_pos_emd, name='sincos_pos_embedding', requires_grad=False)

    def init_weights(self):
        """ ViT weight initialization."""
        for name, cell in self.cells_and_names():
            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
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

    def construct(self, image, unmask_index):
        """construct of VisionTransformerForMae Encoder"""
        # patch to encoder tokens and add positions
        tokens = self.patch_embed(image)

        encoder_pos_embedding = self.stride_slice(
            self.pos_embed, (0, 1, 0),
            (1, self.pos_embed.shape[1], self.pos_embed.shape[2]),
            (1, 1, 1))

        tokens = self.add(tokens, encoder_pos_embedding)

        # get the unmasked tokens to be encoded
        unmask_index_ = self.expand_dim(unmask_index, -1)
        unmask_index = self.tile(unmask_index_, (1, 1, tokens.shape[2]))
        unmask_tokens = self.gather(tokens, 1, unmask_index)

        # cls_tokens add pos_embedding
        cls_pos_embedding = self.stride_slice(
            self.pos_embed, (0, 0, 0),
            (1, 1, self.pos_embed.shape[2]),
            (1, 1, 1))
        batch_size = image.shape[0]
        if batch_size == 1:
            cls_tokens = self.cls_tokens
        else:
            cls_tokens = self.tile(self.cls_tokens, (batch_size, 1, 1))
        cls_tokens = self.add(cls_tokens, cls_pos_embedding)

        # concat cls_tokens
        encoded_tokens = self.cat((cls_tokens, unmask_tokens))
        # attend with vision transformer
        encoder_input_mask = P.Ones()((batch_size, self.seq_length, self.seq_length), mstype.float32)
        for block in self.blocks:
            encoded_tokens = block(encoded_tokens, encoder_input_mask)

        encoded_tokens = self.norm(encoded_tokens)
        return encoded_tokens


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTMAEForPreTraining(MAEPreTrainedModel):
    """
    Pretrain MAE Module.
    The supported model name could be selected from ViTMAEConfig.show_support_list().

    Args:
        config (ViTMAEConfig): the config of Mae model.

    Examples:
        >>> from mindformers import ViTMAEForPreTraining
        >>> model_a = ViTMAEForPreTraining.from_pretrained('mae_vit_base_p16')
        <class 'mindformers.models.mae.mae.ViTMAEForPreTraining'>
        >>> # input config
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('mae_vit_base_p16')
        >>> model_b = ViTMAEForPreTraining(config)
        <class 'mindformers.models.mae.mae.ViTMAEForPreTraining'>
    """
    _support_list = MindFormerBook.get_model_support_list()['mae']

    def __init__(self, config=None):
        config = config if config else ViTMAEConfig()
        super().__init__(config)
        self.use_moe = (config.moe_config.expert_num > 1)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.vit = ViTMAEModel(config)
        self.num_patches = num_patches = self.vit.patch_embed.num_patches
        self.num_masked = num_patches - self.vit.seq_length + 1

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = Linear(
            config.embed_dim, config.decoder_embed_dim, weight_init="xavier_uniform",
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.decoder_embed.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.mask_tokens = Parameter(
            weight_init.initializer(weight_init.Normal(sigma=config.initializer_range),
                                    (1, 1, config.decoder_embed_dim)),
            name='mask_tokens', requires_grad=True)
        self.decoder_pos_embed = Parameter(
            weight_init.initializer(weight_init.Zero(), (1, num_patches + 1, config.decoder_embed_dim)),
            name='pos_embedding', requires_grad=False)
        hdr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.decoder_depth)]
        self.decoder_blocks = nn.CellList([
            Block(hidden_size=config.decoder_embed_dim,
                  ffn_hidden_size=config.decoder_intermediate_size,
                  seq_length=num_patches + 1,
                  drop_rate=config.drop_rate,
                  attention_dropout_rate=config.attention_dropout_rate,
                  layer_norm_eps=config.layer_norm_eps,
                  hidden_dropout_rate=hdr[i],
                  qkv_bias=config.qkv_bias,
                  init_values=config.init_values,
                  weight_init='XavierUniform',
                  layernorm_compute_type=config.layernorm_compute_type,
                  softmax_compute_type=config.softmax_compute_type,
                  window_size=config.window_size,
                  num_heads=config.decoder_num_heads,
                  hidden_act=config.hidden_act,
                  post_layernorm_residual=config.post_layernorm_residual,
                  param_init_type=config.param_init_type,
                  parallel_config=parallel_config.dp_mp_config)
            for i in range(config.decoder_depth)])
        self.decoder_norm = LayerNorm((config.decoder_embed_dim,), eps=config.layer_norm_eps).shard(((dp, 1, 1),))
        patch_dim = config.in_chans * config.patch_size ** 2
        self.decoder_pred = Linear(
            config.decoder_embed_dim, patch_dim, weight_init="xavier_uniform",
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.decoder_pred.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        # --------------------------------------------------------------------------

        self.patchify = Patchify(patch_size=config.patch_size, parallel_config=parallel_config)
        self.unpatchify = UnPatchify(
            patch_size=config.patch_size, seq_length=num_patches, parallel_config=parallel_config)

        self.stride_slice = P.StridedSlice().shard(((1, 1, 1),))
        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.gather = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.gather1 = P.GatherD().shard(((dp, 1), (dp, 1)))
        self.gather2 = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mse_loss = MSELoss(config.norm_pixel_loss, parallel_config)

        self.images_summary = P.ImageSummary().shard(((dp, 1, 1, 1),))

        self.init_weights()
        self.init_pos_emd()

        self.load_checkpoint(config)

    def init_pos_emd(self):
        """init values of pos_embed"""
        decoder_pos_embed = Tensor(
            get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        decoder_pos_embed = P.ExpandDims()(decoder_pos_embed, 0)
        self.decoder_pos_embed = Parameter(decoder_pos_embed, name='decoder_pos_embed', requires_grad=False)

    def init_weights(self):
        """ ViT weight initialization."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
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

    def construct(self, image, mask, ids_restore, unmask_index):
        """construct of VisionTransformerForMae"""
        self.images_summary("input images", image)

        encoder_tokens = self.vit(image, unmask_index)

        unmask_tokens = self.decoder_embed(encoder_tokens)
        unmask_tokens = self.cast(unmask_tokens, mstype.float32)

        # mask tokens add the positions using the masked indices derived above
        batch_size = encoder_tokens.shape[0]
        mask_tokens = self.tile(self.mask_tokens, (batch_size, self.num_masked, 1))

        # concat the masked tokens to the decoder tokens and attend with decoder
        img_tokens = self.stride_slice(
            unmask_tokens, (0, 1, 0),
            (unmask_tokens.shape[0], unmask_tokens.shape[1], unmask_tokens.shape[2]), (1, 1, 1))
        full_tokens_ = self.cat((img_tokens, mask_tokens))
        ids_restore_copy = ids_restore
        ids_restore_ = self.expand_dim(ids_restore_copy, -1)
        ids_restore_ = self.tile(ids_restore_, (1, 1, unmask_tokens.shape[2]))
        full_tokens_ = self.gather2(full_tokens_, 1, ids_restore_)
        cls_tokens = self.stride_slice(
            unmask_tokens, (0, 0, 0),
            (unmask_tokens.shape[0], 1, unmask_tokens.shape[2]), (1, 1, 1))
        decoder_tokens = self.cat((cls_tokens, full_tokens_))

        # add position embendding for decoder tokens
        decoder_tokens = self.add(decoder_tokens, self.decoder_pos_embed)
        # decoder
        attention_mask = Tensor(np.ones((batch_size, self.num_patches + 1, self.num_patches + 1)), mstype.float32)
        for block in self.decoder_blocks:
            decoder_tokens = block(decoder_tokens, attention_mask)

        # normalize decoder tokens
        decoder_tokens = self.decoder_norm(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        pred = self.decoder_pred(decoder_tokens)
        pred = self.cast(pred, mstype.float32)

        pred = self.stride_slice(pred, (0, 1, 0), (pred.shape[0], pred.shape[1], pred.shape[2]), (1, 1, 1))

        reconstruct_images = self.unpatchify(pred)
        self.images_summary("reconstruct image", reconstruct_images)

        if not self.training:
            return reconstruct_images

        patches = self.patchify(image)
        mask = self.gather1(mask, 1, ids_restore)
        mae_loss = self.mse_loss(pred, patches, mask)
        return mae_loss
