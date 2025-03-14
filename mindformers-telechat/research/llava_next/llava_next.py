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
"""
Llava Model
"""
import math

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers import BaseXModalToTextModel
from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.build_model import build_network
from mindformers.modules.activation import GELU
from mindformers.models.utils import lazy_inline
from mindformers.modules.layers import Linear
from mindformers.tools import logger
from research.llava_next.llava_next_config import LlavaNextConfig


class LlavaNextPooler(nn.Cell):
    r"""
    Module to pooler dimension
    Args:
        config (LlavaNextConfig): The config of Llava next model.
    """

    def __init__(self, config: LlavaNextConfig, **kwargs):
        super(LlavaNextPooler, self).__init__(config, **kwargs)
        mode = config.spatial_pool_mode
        self.mode = mode
        stride = config.spatial_pool_stride
        self.pooler_dtype = ms.float32
        self.stride = stride
        self.reshape = P.Reshape()
        self.height = int(math.sqrt(config.num_queries))
        self.width = self.height
        out_channels = getattr(config, "spatial_pool_out_channels", config.vision_model.model_config.hidden_size)
        self.transpose = P.Transpose()
        self.transpose_2 = P.Transpose()
        self.image_size = config.vision_model.model_config.image_size // \
                          config.vision_model.model_config.patch_size ** 2
        self.cast = P.Cast()
        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        elif mode == "conv":
            self.pool = nn.Conv2d(
                in_channels=config.vision_model.model_config.hidden_size,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
            )
        elif mode == "trilinear":
            # mindspore support trilinear
            self.pool = P.UpsampleTrilinear3D()
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Has to be one of [`average`, `max`, `conv`,`trilinear`]")

    def construct(self, image_features):
        """pooler forward method"""
        ori_dtype = image_features.dtype
        ori_shape = image_features.shape
        image_features = self.reshape(image_features, (ori_shape[0], self.height, self.width, ori_shape[2]))
        image_features = self.transpose(image_features, (0, 3, 1, 2))
        image_features = self.cast(image_features, self.pooler_dtype)
        if self.mode != 'trilinear':
            image_features = self.pool(image_features)
        else:
            image_features = self.reshape(image_features, (ori_shape[0], ori_shape[2], 1, self.height, self.width))
            output_size = [1, math.ceil(self.height / self.stride), math.ceil(self.width / self.stride)]
            image_features = self.pool(image_features, output_size, None)
            _, _, _, linear_weight, linear_width = image_features.shape
            image_features = self.reshape(image_features, (ori_shape[0], ori_shape[2], linear_weight, linear_width))
        image_features = self.cast(image_features, ori_dtype)
        new_shape = image_features.shape
        image_features = self.reshape(image_features, (new_shape[0], new_shape[1], new_shape[2] * new_shape[3]))
        image_features = self.transpose_2(image_features, (0, 2, 1))

        return image_features

    def shard(self, parallel_config):
        """distributed shard config"""
        dp = parallel_config.data_parallel
        self.transpose.shard(((dp, 1, 1, 1),))
        self.transpose_2.shard(((dp, 1, 1),))
        if self.mode == 'average':
            self.pool.avg_pool.shard(((dp, 1, 1, 1),))
        elif self.mode == "max":
            self.pool.max_pool.shard(((dp, 1, 1, 1),))
        elif self.mode == "conv":
            self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
            self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
        elif self.mode == "trilinear":
            self.pool.shard(((dp, 1, 1, 1, 1, 1),))


class LlavaAdapter(nn.Cell):
    r"""
    Module to resize the image dimension into the llm model dimension
    Args:
        config (LlavaNextConfig): The config of Llava model.
    """

    def __init__(self, config: LlavaNextConfig):
        super(LlavaAdapter, self).__init__(config)

        self.adapter = Linear(
            in_channels=config.vision_model.model_config.hidden_size,
            out_channels=config.text_model.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )
        self.activation_func = GELU(approximate=False)
        self.adapter_2 = Linear(
            in_channels=config.text_model.model_config.hidden_size,
            out_channels=config.text_model.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )
        self.dtype = P.DType()
        self.cast = P.Cast()

    def construct(self, x):
        """adapter forward method"""
        output = self.adapter(x)
        ori_type = self.dtype(output)
        output = self.cast(output, mstype.float32)
        output = self.activation_func(output)
        output = self.cast(output, ori_type)
        output = self.adapter_2(output)
        return output

    def shard(self, parallel_config):
        """distributed method"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.adapter.shard(((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.adapter_2.shard(((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlavaNextVlm(BaseXModalToTextModel):
    """
    Provide Llava1.5 training loss or logits through network.

    Args:
        config (LlavaNextConfig): The config of Llava model.
    """

    @lazy_inline
    def __init__(self, config: LlavaNextConfig, **kwargs):
        super(LlavaNextVlm, self).__init__(config, **kwargs)
        self.config = config if config is not None else LlavaNextConfig()
        self.language_model = build_network(self.config.text_model)
        self.vision_encoder = build_network(self.config.vision_model)
        self.adapter = LlavaAdapter(self.config)
        self.add_newline = config.add_newline
        self.dtype = config.compute_dtype
        if config.add_newline:
            scale = 1 / math.sqrt(config.text_model.model_config.hidden_size)
            self.image_newline = Parameter(
                scale * Tensor(np.random.normal(0, 1, size=(config.text_model.model_config.hidden_size))).astype(
                    config.param_init_type))
        self.max_patch_height_num = config.max_patch_height_num
        self.max_patch_width_num = config.max_patch_width_num
        self.vision_encoder.pipeline_stage = 0
        self.adapter.pipeline_stage = 0

        self.is_first_iteration = True
        self.use_past = config.use_past
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.image_size = self.config.vision_model.model_config.image_size
        self.num_queries = self.config.vision_model.model_config.num_queries
        self.ignore_token_id = Tensor(config.ignore_token_id, mstype.int32)
        self.height = int(math.sqrt(config.vision_model.model_config.num_queries))
        self.width = self.height
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.slice = P.StridedSlice()
        self.slice_img = P.StridedSlice()
        self.transpose = P.Transpose()
        self.tile = P.Tile()
        self.concat_img = P.Concat(axis=3)
        self.concat_img_new = P.Concat(axis=1)
        self.concat_patch = P.Concat(axis=1)
        self.transpose_img = P.Transpose()
        self.cast = P.Cast()
        self.video_contains_pooler = self.config.video_contains_pooler

        self.training_stage = f"stage_{str(config.stage)}"
        if self.config.video_contains_pooler and self.training_stage == "stage_3":
            self.pooler = LlavaNextPooler(config)
            self.pooler.shard(config.parallel_config)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.adapter.shard(config.parallel_config)
            self.slice.shard(((dp, 1),))
            self.slice_img.shard(((dp, 1, 1, 1),))
            self.transpose.shard(((dp, 1, 1, 1, 1, 1),))
            self.tile.shard(((1, 1, 1, 1),))
            self.concat_img.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.concat_img_new.shard(((dp, 1, 1), (dp, 1, 1)))
            self.concat_patch.shard(((dp, 1, 1, 1, 1), (dp, 1, 1, 1, 1)))
            self.transpose_img.shard(((dp, 1, 1),))
            if self.training_stage != "stage_3":
                if config.img_dynamic_batch:
                    from mindspore import Layout
                    self.tensor_scatter_update = P.TensorScatterUpdate().add_prim_attr("self_define_shard", True)
                    layout = Layout((dp, 1, 1, mp), ("dp", "cp", "sp", "mp"))
                    layout_tuple = (
                        layout("dp", "cp", "mp"), layout("dp", "cp", "sp"), layout("dp", "cp", "mp"))
                    self.tensor_scatter_update.shard(layout_tuple, out_strategy=(layout("dp", "cp", "mp"),))
                else:
                    self.tensor_scatter_update.shard(((1, 1, mp), (1, 1, 1), (1, 1, mp)))
        self.freeze_component()

    def freeze_component(self):
        """freeze parameters of model according to different command"""
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_encoder.trainable_params():
                param.requires_grad = False

        if self.config.freeze_llm:
            logger.info("freeze llm model")
            for name, param in self.parameters_and_names():
                if "adapter" not in name and "vision_encoder" not in name:
                    param.requires_grad = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation in inference"""
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
        images = kwargs.pop("images")
        image_context_pos = kwargs.pop("image_context_pos", None)
        image_patches = kwargs.pop("image_patches", None)
        slot_mapping = kwargs.get("slot_mapping")
        if image_context_pos is not None and not isinstance(image_context_pos, ms.Tensor):
            image_context_pos = ms.Tensor(image_context_pos, mstype.int32)
        if image_patches is not None and not isinstance(image_patches, ms.Tensor):
            image_patches = ms.Tensor(image_patches, mstype.float32)
        if image_patches is not None:
            model_inputs = {
                "input_ids": ms.Tensor(input_ids, mstype.int32),
                "images": images,
                "image_context_pos": image_context_pos,
                "image_patches": image_patches,
            }

        else:
            model_inputs = {
                "input_ids": ms.Tensor(input_ids, mstype.int32),
                "images": images,
                "image_context_pos": image_context_pos
            }
        if kwargs.get("prefill"):
            batch_valid_length = kwargs.get("valid_length_each_example")
            model_inputs = self._prepare_inputs_for_prefill_flatten(input_ids, batch_valid_length, slot_mapping,
                                                                    model_inputs)
        return model_inputs

    def _prepare_inputs_for_prefill_flatten(self, input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]  # [bs,]
        input_ids_list = []
        for i in range(batch_valid_length_bs):
            context_len = batch_valid_length[i]
            input_ids_list.append(input_ids[i][:context_len])
        input_ids = np.concatenate(input_ids_list, 0)
        input_ids = input_ids.reshape((1, -1))
        slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        return model_inputs

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """prepare inputs for predict layout"""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq = input_ids.shape
        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((bs, 1, 3, self.image_size, self.image_size)), ms.float32)

        if "image_context_pos" in kwargs:
            image_context_pos = Tensor(kwargs.get("image_context_pos"))
        else:
            if self.training_stage == "stage_3":
                num_queries = images.shape[1] * self.num_queries // 4
                image_context_pos = Tensor(np.random.randint(0, num_queries, (bs, 1, num_queries, 2)),
                                           ms.int32)
                image_patches = None
            else:
                num_queries = 2928
                image_context_pos = Tensor(np.random.randint(0, num_queries, (bs, 1, num_queries, 2)),
                                           ms.int32)
                image_patches = Tensor(np.random.random((bs, 2, 2, 3, self.image_size, self.image_size)),
                                       ms.float32)

        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, images, image_patches, image_context_pos, None, None, None, None, None, None, None, None, \
               None, slot_mapping

    def set_dynamic_inputs(self):
        """set inputs when is_dynamic=True"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None, None], dtype=mstype.float32)
        dynamic_image_patches = Tensor(shape=[None, None, None, None, None, None], dtype=mstype.float32) \
            if self.training_stage != "stage_3" else None
        dynamic_img_pos = Tensor(shape=[None, None, None, 2], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_image_patches, dynamic_img_pos, None, None, None,
                        None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        logger.info("Set dynamic inputs for LLava-Next")

    def sing_image_construct(self, image_embeds, image_patches_shape, original_shape):
        """ single image with any res construct"""
        image_embeds = self.cast(image_embeds, self.dtype)
        _, image_seq, image_dim = image_embeds.shape
        if image_patches_shape is not None:
            image_embeds = self.reshape(image_embeds, (original_shape[0], original_shape[1], image_seq, image_dim))
            image_patches = self.slice_img(image_embeds, (0, 1, 0, 0),
                                           (original_shape[0], original_shape[1], image_seq, image_dim), (1, 1, 1, 1))
            base_image = self.slice_img(image_embeds, (0, 0, 0, 0), (original_shape[0], 1, image_seq, image_dim),
                                        (1, 1, 1, 1))

            image_patches = self.reshape(image_patches,
                                         (original_shape[0], image_patches_shape[1], image_patches_shape[2],
                                          self.height, self.width, image_dim))

            image_patches = self.transpose(image_patches,
                                           (0, 5, 1, 3, 2, 4)).contiguous()
            image_patches = self.reshape(image_patches, (original_shape[0], image_dim,
                                                         image_patches_shape[1] * self.height,
                                                         image_patches_shape[2] * self.width))
            if self.add_newline:
                image_newline = self.cast(self.reshape(self.image_newline, (1, -1, 1, 1)), self.dtype)
                _, _, processed_height, _ = image_patches.shape
                image_newline = self.tile(image_newline, (original_shape[0], 1, processed_height, 1))
                image_patches = self.concat_img([image_patches, image_newline])
            _, _, processed_height, processed_width = image_patches.shape
            image_patches = image_patches.reshape((original_shape[0], image_dim, processed_height * processed_width))
            image_patches = self.transpose_img(image_patches, (0, 2, 1))
            base_image_shape = base_image.shape
            base_image = self.reshape(base_image, (base_image_shape[0] * base_image_shape[1],
                                                   base_image_shape[2],
                                                   base_image_shape[3]))
            image_embeds = self.concat_img_new([image_patches, base_image])

        return image_embeds

    def video_construct(self, image_embed, original_shape):
        """video forward method"""
        if self.video_contains_pooler:
            image_embed = self.pooler(image_embed)
        pooler_shape = image_embed.shape
        image_embed = self.reshape(image_embed,
                                   (original_shape[0], original_shape[1] * pooler_shape[1], pooler_shape[2]))
        return image_embed

    def construct(self, input_ids, images, image_patches: Tensor = None, image_context_pos: Tensor = None, labels=None,
                  input_position=None, position_ids=None,
                  attention_mask=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        r"""
        LlavaNextVlm forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            images(Tensor): the image tensor with datatype float32, Tensor of shape :math:
            `(batch, 1, image_number, channel, image_resolution, image_resolution)`
            image_context_pos(Tensor): the position index of the image in final input embedding. Tensor of shape :math
            `(batch,1, num_queries, 2)`
            image_patches(Tensor): The patched images from single image for any res method. Tensor of shape :math:
            `(batch, patched_height, patched_weight, channel, image_resolution, image_resolution)`
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): the input embedding Tensor of shape :math:`(batch, seq\_length, hidden_size)`.
                Default None.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """

        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
        else:
            tokens = input_ids
        input_embeds = self.language_model.to_embeddings(tokens)

        if self.is_first_iteration:
            images_shape = self.shape(images)
            # bs 1 3 336 336
            image_patches_shape = None
            if image_patches is not None:
                image_patches_shape = image_patches.shape
                image_patches = self.reshape(image_patches,
                                             (image_patches_shape[0], image_patches_shape[1] * image_patches_shape[2],
                                              image_patches_shape[3], image_patches_shape[4], image_patches_shape[5]))
                images = self.concat_patch([images, image_patches])
                images_shape = self.shape(images)
            new_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
            images = self.reshape(images, new_shape)
            images_pos_shape = self.shape(image_context_pos)
            new_shape = (images_pos_shape[0] * images_pos_shape[1], images_pos_shape[2], images_pos_shape[3])
            image_context_pos = self.reshape(image_context_pos, new_shape)
            image_embeds = self.vision_encoder(images)  # [bs*frames, 576, 8192]
            image_embeds = self.adapter(image_embeds)  # bs * frames, 576, 8192
            if self.training_stage == 'stage_3':
                image_embeds = self.video_construct(image_embeds, images_shape)
            else:
                image_embeds = self.sing_image_construct(image_embeds, image_patches_shape, images_shape)

            input_embeds = self.update_modal_to_text(image_embeds, input_embeds, image_context_pos)

        return self.language_model(
            input_ids=input_ids,
            input_embeds=input_embeds,
            labels=labels,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            input_position=input_position,
            init_reset=init_reset,
            position_ids=position_ids,
            attention_mask=attention_mask,
            batch_index=batch_index,
            zactivate_len=zactivate_len
        )

    def kvcache(self, layer_idx):
        key_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
