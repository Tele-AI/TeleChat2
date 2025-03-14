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
Internvl Model
"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P

from mindformers.tools import logger
from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models import build_network
from mindformers.models.utils import lazy_inline
from mindformers.modules.layers import LayerNorm, Linear
from mindformers.models.multi_modal.base_model import BaseXModalToTextModel
from research.internvl2.internvl_configuration import InternVLChatConfig


class MLP(nn.Cell):
    r"""
    Module to resize the image dimension into the llm model dimension
    Args:
        config (LlavaConfig): The config of Llava model.
    """

    def __init__(self, config: InternVLChatConfig):
        super(MLP, self).__init__(config)
        vision_config = config.vision_model.model_config
        text_config = config.text_model.model_config

        self.layer_norm = LayerNorm(vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.adapter1 = Linear(
            in_channels=vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2,
            out_channels=text_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )
        self.activation_func = P.GeLU()
        self.adapter2 = Linear(
            in_channels=text_config.hidden_size,
            out_channels=text_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.adapter1.shard(((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.adapter2.shard(((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))
        self.activation_func.shard(((dp, 1, mp),))

    def construct(self, x):
        """adapter forward method"""
        output = self.layer_norm(x)
        output = self.adapter1(output)
        output = self.activation_func(output)
        output = self.adapter2(output)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InternVLChatModel(BaseXModalToTextModel):
    """
    Provide InternVL2 training loss or logits through network.

    Args:
        config (InternVLChatConfig): The config of InternVL model.
    """
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer']

    @lazy_inline
    def __init__(self, config: InternVLChatConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.vision_config = config.vision_model.model_config
        self.text_config = config.text_model.model_config

        image_size = config.force_image_size or self.vision_config.image_size
        patch_size = self.vision_config.patch_size
        dp = config.parallel_config.data_parallel
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.image_size = image_size

        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.select_layer = config.select_layer
        self.num_queries = config.num_queries

        self.base_index_adder = None
        self.tensor_scatter_update = P.TensorScatterUpdate()
        self.expand_dims = P.ExpandDims()

        self.vision_model = build_network(config.vision_model)
        self.language_model = build_network(config.text_model)

        self.img_context_token_id = config.img_context_token_id
        self.img_pos_add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.vit_hidden_size = self.vision_config.hidden_size
        self.text_hidden_size = self.text_config.hidden_size
        self.mlp1 = MLP(config)

        self.is_first_iteration = True

        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.add = P.Add().shard(((dp, 1), ()))
        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.ones = P.Ones()
        self.gather = P.Gather(1).shard(((dp, 1, 1), (dp,)))
        self.prefill_gather_flatten = P.Gather().shard(((dp, 1, 1), (dp,)))
        self.sub_batch_valid_len = P.Sub().shard(((1,), ()))
        self.is_first_iteration = True
        self.transpose_4dim = P.Transpose().shard(((dp, 1, 1, 1),))
        self.slice_3dim = P.StridedSlice().shard(((dp, 1, 1),))

    def set_dynamic_inputs(self):
        """set inputs when is_dynamic=True"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
        dynamic_img_pos = Tensor(shape=[None, self.num_queries, 2], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_img_pos, None, None, None,
                        None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        logger.info("Set dynamic inputs for Internvl2")

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """prepare inputs for predict layout"""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq = input_ids.shape
        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((13, 3, self.image_size, self.image_size)), ms.float32)

        if "image_context_pos" in kwargs:
            image_context_pos = Tensor(kwargs.get("image_context_pos"))
        else:
            image_context_pos = Tensor(np.ones((13, self.num_queries, 2)), ms.int32)

        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, images, image_context_pos, None, None, None, None, None, None, None, None, None, slot_mapping

    def _prepare_inputs_for_prefill_flatten(self, input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]
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

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation in inference"""
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
        images = kwargs.pop("images")
        slot_mapping = kwargs.pop("slot_mapping")
        image_context_pos = kwargs.pop("image_context_pos", None)
        if image_context_pos is not None and not isinstance(image_context_pos, ms.Tensor):
            image_context_pos = ms.Tensor(image_context_pos, mstype.int32)

        if slot_mapping is not None:
            model_inputs = {
                "input_ids": ms.Tensor(input_ids, mstype.int32),
                "images": images,
                "image_context_pos": image_context_pos,
                "slot_mapping": Tensor.from_numpy(slot_mapping)
            }

            if self.is_first_iteration:
                batch_valid_length = kwargs.get("valid_length_each_example")
                model_inputs = self._prepare_inputs_for_prefill_flatten(
                    input_ids, batch_valid_length, slot_mapping, model_inputs
                )
        else:
            model_inputs = {
                "input_ids": ms.Tensor(input_ids, mstype.int32),
                "images": images,
                "image_context_pos": image_context_pos
            }
        return model_inputs

    def pixel_shuffle(self, x, scale_factor=0.5):
        """shuffle pixel tensor"""
        n, w, h, c = self.shape(x)
        new_h = int(h * scale_factor)
        new_c = int(c / scale_factor)
        x = self.reshape(x, (n, w, new_h, new_c))
        x = self.transpose_4dim(x, (0, 2, 1, 3))
        new_w = int(w * scale_factor)
        new_c = int(c / (scale_factor * scale_factor))
        x = self.reshape(x, (n, new_h, new_w, new_c))
        x = self.transpose_4dim(x, (0, 2, 1, 3))
        return x

    def extract_feature(self, pixel_values):
        """get image embeds"""
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            select_layer=self.select_layer)
        old_vit_embeds_shape = self.shape(vit_embeds)
        vit_embeds = self.slice_3dim(vit_embeds,
                                     (0, 1, 0),
                                     (old_vit_embeds_shape[0], old_vit_embeds_shape[1], old_vit_embeds_shape[2]),
                                     (1, 1, 1))
        vit_embeds_shape = self.shape(vit_embeds)
        h = int(vit_embeds_shape[1] ** 0.5)
        w = int(vit_embeds_shape[1] ** 0.5)
        vit_embeds = self.reshape(vit_embeds, (vit_embeds_shape[0], w, h, vit_embeds_shape[2]))

        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds_shape_2 = self.shape(vit_embeds)
        vit_embeds = self.reshape(
            vit_embeds, (vit_embeds_shape_2[0], vit_embeds_shape_2[1] * vit_embeds_shape_2[2], vit_embeds_shape_2[3]))
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def construct(self, input_ids, images, image_context_pos: Tensor = None, labels=None, input_position=None,
                  position_ids=None, attention_mask=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        """forward of InternVL"""
        tokens = input_ids

        # Convert input tokens to embeddings
        input_embeds = self.language_model.to_embeddings(tokens)
        if self.is_first_iteration:
            image_embeds = self.extract_feature(images)
            input_embeds = self.update_modal_to_text(image_embeds, input_embeds, image_context_pos)

        return self.language_model(
            input_ids=input_ids,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            slot_mapping=slot_mapping,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            init_reset=init_reset,
            batch_index=batch_index,
            zactivate_len=zactivate_len
        )
