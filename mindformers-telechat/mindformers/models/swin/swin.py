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
# https://github.com/microsoft/Swin-Transformer
# ============================================================================
"""Swin Model."""
import os

import numpy as np
from scipy import interpolate

from mindspore import nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import dtype as mstype
import mindspore.ops.operations as P
import mindspore.common.initializer as weight_init_
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from mindformers.tools.logger import logger
from mindformers.tools.download_tools import download_with_progress_bar
from mindformers.core.loss import build_loss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.swin.swin_config import SwinConfig
from mindformers.models.swin.swin_modules import Linear
from mindformers.models.swin.swin_modules import LayerNorm
from mindformers.models.swin.swin_modules import Dropout
from mindformers.models.swin.swin_modules import SwinPatchEmbeddings
from mindformers.models.swin.swin_modules import SwinPatchMerging
from mindformers.models.swin.swin_modules import SwinStage
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import try_sync_file


__all__ = ['SwinForImageClassification', 'SwinModel']


class SwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SwinConfig
    base_model_prefix = "swin"


@MindFormerRegister.register(MindFormerModuleType.ENCODER)
class SwinBaseModel(SwinPreTrainedModel):
    """
    An abstract class to handle weights initialization and save weights decay grouping.
    """

    def init_weights(self):
        """ Swin weight initialization, original timm impl (for reproducibility) """
        for _, cell in self.cells_and_names():
            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init_.initializer(
                    weight_init_.TruncatedNormal(sigma=0.02),
                    cell.weight.shape,
                    cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                                cell.bias.shape,
                                                                cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init_.initializer(weight_init_.One(),
                                                             cell.gamma.shape,
                                                             cell.gamma.dtype))
                cell.beta.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                            cell.beta.shape,
                                                            cell.beta.dtype))

    @staticmethod
    def no_weight_decay():
        return {'absolute_pos_embed'}

    @staticmethod
    def no_weight_decay_keywords():
        return {'relative_position_bias_table'}


# pylint: disable=C0326
@MindFormerRegister.register(MindFormerModuleType.ENCODER)
class SwinModel(SwinBaseModel):
    """
    Swin Transformer.
    The supported model name could be selected from SwinForImageClassification.show_support_list().

    Args:
        config (SwinConfig): the config of Swin model.
    """

    def __init__(self, config: PretrainedConfig = None):
        if config is None:
            config = SwinConfig()
        super(SwinModel, self).__init__(config)
        dp = config.parallel_config.data_parallel
        self.parallel_config = config.parallel_config
        self.use_moe = config.moe_config.expert_num > 1
        self.num_labels = config.num_labels
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.ape = config.use_absolute_embeddings
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.patch_norm = config.patch_norm
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = config.mlp_ratio
        self.cast = P.Cast()

        # split image into non-overlapping patches
        self.patch_embed = SwinPatchEmbeddings(config)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(
                Tensor(np.zeros((1, num_patches, config.embed_dim)), dtype=mstype.float32), name="ape")

        self.pos_drop = Dropout(keep_prob=1.0 - config.hidden_dropout_prob)
        self.pos_drop.shard(((dp, 1, 1),))

        # stochastic depth
        dpr = list(np.linspace(0, config.drop_path_rate, sum(config.depths)))  # stochastic depth decay rule
        if self.use_moe:
            parallel_config_args = config.parallel_config.moe_parallel_config
        else:
            parallel_config_args = config.parallel_config.dp_mp_config

        # build layers
        self.layers = nn.CellList()
        self.final_seq = num_patches  # downsample seq_length
        for i_layer in range(self.num_layers):
            layer = SwinStage(
                config=config,
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=LayerNorm,
                downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                parallel_config=parallel_config_args)
            # downsample seq_length
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)
        self.norm = LayerNorm([self.num_features, ], eps=config.layer_norm_eps).shard(((dp, 1, 1), ))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.avgpool = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1), ))
        self.init_weights()

    def construct(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(self.transpose(x, (0, 2, 1)), 2)  # B C 1
        return x


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class SwinForImageClassification(SwinBaseModel):
    """
    Swin Transformer Model.
    The supported model name could be selected from SwinForImageClassification.show_support_list().

    Args:
        config (SwinConfig): the config of Swin model.

    Examples:
        >>> from mindformers import SwinForImageClassification, AutoConfig
        >>> # input model name, load model and weights
        >>> model_a = SwinForImageClassification.from_pretrained('swin_base_p4w7')
        >>> type(model_a)
        <class 'mindformers.models.swin.swin.SwinForImageClassification'>
        >>> # input config, load model without weights
        >>> config = AutoConfig.from_pretrained('swin_base_p4w7')
        >>> model_b = SwinForImageClassification(config)
        >>> type(model_b)
        <class 'mindformers.models.swin.swin.SwinForImageClassification'>
    """
    _support_list = MindFormerBook.get_model_support_list()['swin']

    def __init__(self, config: PretrainedConfig = None):
        if config is None:
            config = SwinConfig()
        super(SwinForImageClassification, self).__init__(config)
        self.encoder = SwinModel(config)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.head = Linear(
            self.encoder.num_features, self.encoder.num_labels,
            weight_init=weight_init_.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)
        self.head.shard(strategy_bias=((dp, mp), (mp,)), strategy_matmul=((dp, 1), (mp, 1)))

        self.loss = build_loss(class_name=config.loss_type)

        self.load_checkpoint(config)

    def construct(self, image, target=None):
        x = self.encoder(image)
        out = self.head(x)
        if not self.training:
            return out, target
        loss = self.loss(out, target)
        return loss

    def load_checkpoint(self, config):
        """
        load checkpoint for SwinForImageClassification.

        Args:
            config (ModelConfig): a model config instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        if checkpoint_name_or_path:
            if not isinstance(checkpoint_name_or_path, str):
                raise TypeError(f"checkpoint_name_or_path should be a str,"
                                f" but got {type(checkpoint_name_or_path)}")

            if os.path.exists(checkpoint_name_or_path):
                param = load_checkpoint(checkpoint_name_or_path)
                ckpt_file = checkpoint_name_or_path
                param = self.remap_pretrained_keys_swin(param)

                try:
                    load_param_into_net(self, param)
                except RuntimeError:
                    logger.error("the given config and weights in %s are"
                                 " mismatched, and weights load failed", ckpt_file)
                logger.info("weights in %s are loaded", ckpt_file)

            elif checkpoint_name_or_path not in self._support_list:
                raise ValueError(f"{checkpoint_name_or_path} is not a supported default model"
                                 f" or a valid path to checkpoint,"
                                 f" please select from {self._support_list}.")
            else:
                default_checkpoint_download_folder = os.path.join(
                    MindFormerBook.get_default_checkpoint_download_folder(), checkpoint_name_or_path.split("_")[0])
                if not os.path.exists(default_checkpoint_download_folder):
                    os.makedirs(default_checkpoint_download_folder, exist_ok=True)
                ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name_or_path + ".ckpt")
                if not os.path.exists(ckpt_file):
                    url = MindFormerBook.get_model_ckpt_url_list()[checkpoint_name_or_path][0]
                    succeed = download_with_progress_bar(url, ckpt_file)
                    if not succeed:
                        logger.info("checkpoint download failed, and pretrained weights are unloaded.")
                        return
                try_sync_file(ckpt_file)

                logger.info("start to read the ckpt file: %s", os.path.getsize(ckpt_file))
                param = load_checkpoint(ckpt_file)
                try:
                    load_param_into_net(self, param)
                except RuntimeError:
                    logger.error("the given config and weights in %s are"
                                 " mismatched, and weights load failed", ckpt_file)
                logger.info("weights in %s are loaded", ckpt_file)
        else:
            logger.info("model built, but weights is unloaded, since the config has no"
                        " checkpoint_name_or_path attribute or"
                        " checkpoint_name_or_path is None.")

    def remap_pretrained_keys_swin(self, checkpoint_model):
        # This class was refer to project:
        # https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py
        """remap pretrained keys swin"""
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        state_dict = self.parameters_dict()
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_bias_table" in key and "adam" not in key:
                relative_position_bias_table_pretrained = checkpoint_model[key]
                relative_position_bias_table_current = state_dict[key]
                l1, n_h1 = relative_position_bias_table_pretrained.shape
                l2, n_h2 = relative_position_bias_table_current.shape
                if n_h1 != n_h2:
                    logger.info("Error in loading %s, passing......", key)
                else:
                    if l1 != l2:
                        logger.info("%s: Interpolate relative_position_bias_table using geo.", key)
                        src_size = int(l1 ** 0.5)
                        dst_size = int(l2 ** 0.5)

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0

                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)

                        str_x = str(x)
                        str_dx = str(dx)

                        logger.info("Original positions = %s", str_x)
                        logger.info("Target positions = %s", str_dx)

                        all_rel_pos_bias = []

                        for i in range(n_h1):
                            z = relative_position_bias_table_pretrained[:, i].view(
                                src_size, src_size).asnumpy().astype(np.float32)
                            f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(Tensor(f_cubic(dx, dy), mstype.float32).view(-1, 1))

                        new_rel_pos_bias = P.Concat(axis=-1)(tuple(all_rel_pos_bias))
                        new_rel_pos_bias = Parameter(new_rel_pos_bias)
                        checkpoint_model[key] = new_rel_pos_bias

        # delete relative_coords_table and attn_mask, since we always re-init it
        checkpoint_model = self._del_unused_keys(checkpoint_model, "relative_position_index")
        checkpoint_model = self._del_unused_keys(checkpoint_model, "relative_coords_table")
        checkpoint_model = self._del_unused_keys(checkpoint_model, "attn_mask")
        return checkpoint_model

    @staticmethod
    def _del_unused_keys(checkpoint_model, key_name):
        unwanted_key = [k for k in checkpoint_model.keys() if key_name in k]
        for k in unwanted_key:
            del checkpoint_model[k]
        return checkpoint_model
