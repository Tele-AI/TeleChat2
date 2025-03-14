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
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""layers for visualglm"""

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P

from mindformers.models.configuration_utils import PretrainedConfig


class ImageTextEmbeddingConcat(nn.Cell):
    """
    Layer to concat image embedding and text embedding
    """

    def __init__(self, pad_token_id):
        super().__init__()
        self.concat_2d = P.Concat(axis=1)
        self.concat_3d = P.Concat(axis=1)
        self.not_equal = P.NotEqual()
        self.ones = P.Ones()
        self.cast = P.Cast()

        self.pad_token_id = pad_token_id

    def construct(self, image_embeddings: ms.Tensor, pre_text_embeddings: ms.Tensor, post_text_embeddings: ms.Tensor):
        pre_text_embeddings = self.cast(pre_text_embeddings, mstype.float32)
        post_text_embeddings = self.cast(post_text_embeddings, mstype.float32)
        concat_embeds = self.concat_3d([pre_text_embeddings, image_embeddings, post_text_embeddings])
        return concat_embeds


class ImageTextEmbeddingPreparationMixIn:
    """
    image text embemdding mixin
    """

    def __init__(self, config: PretrainedConfig):
        """init method"""
        pad_token_id = 3 if config.pad_token_id is None else config.pad_token_id
        self.image_text_concat = ImageTextEmbeddingConcat(pad_token_id)

    def to_text_embeddings(self, text_input_ids):
        raise NotImplementedError

    def prepare_image_text_embedding(self, input_ids, **kwargs):
        """ prepare image and text embeddings """
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = ms.Tensor(input_position, mstype.int32)

        concat_inputs_embeds = None
        if self.is_first_iteration or not self.use_past:
            image_embeddings = kwargs.get("image_embeds")
            pre_input_ids = ms.Tensor(kwargs.get("pre_input_ids"), mstype.int32)

            image_embeddings_length = image_embeddings.shape[1]
            pre_text_embeddings_length = pre_input_ids.shape[1]
            post_input_ids = ms.Tensor(input_ids[:, image_embeddings_length + pre_text_embeddings_length:],
                                       mstype.int32)
            pre_text_embeddings = self.to_text_embeddings(pre_input_ids)
            post_text_embeddings = self.to_text_embeddings(post_input_ids)
            concat_inputs_embeds = self.image_text_concat(image_embeddings, pre_text_embeddings, post_text_embeddings)
        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "input_embeddings": concat_inputs_embeds,
            "attention_mask": ms.Tensor(attention_mask, mstype.int32),
            "position_ids": ms.Tensor(position_ids, mstype.int32),
            "input_position": input_position
        }
