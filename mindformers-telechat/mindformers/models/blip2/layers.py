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
"""layers for blip2"""

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

    def construct(self, image_embeddings: ms.Tensor, text_embeddings: ms.Tensor, text_input_ids: ms.Tensor):
        text_embeddings = self.cast(text_embeddings, mstype.float32)
        text_embeddings_atts = self.cast(self.not_equal(text_input_ids, self.pad_token_id), mstype.float32)

        image_embeddings_atts = self.ones(image_embeddings.shape[:-1], mstype.float32)

        concat_embeds = self.concat_3d([image_embeddings, text_embeddings])
        concat_attention_mask = self.concat_2d([image_embeddings_atts, text_embeddings_atts])
        return concat_embeds, concat_attention_mask


class ImageTextEmbeddingPreparationMixIn:
    """
    In image to text generation task, image embedding will be concat to text embedding, then put into llm model inputs.
    It realizes the concat of image and text embeddings for llm model. The llm model need call
    method prepare_image_text_embedding in method prepare_inputs_for_generation and impl to_text_embeddings.
    """
    def __init__(self, config: PretrainedConfig):
        pad_token_id = 0 if config.pad_token_id is None else config.pad_token_id
        self.image_text_concat = ImageTextEmbeddingConcat(pad_token_id)

    def to_text_embeddings(self, text_input_ids):
        raise NotImplementedError

    def prepare_image_text_embedding(self, input_ids, **kwargs):
        """
        concat image embedding and text embedding
        """
        input_position = kwargs.pop("current_index", None)
        if input_position is not None:
            input_position = ms.Tensor(input_position, mstype.int32)

        if self.is_first_iteration or not self.use_past:
            image_embeddings = kwargs.pop("image_embeds")

            image_embeddings_length = image_embeddings.shape[1]
            text_input_ids = ms.Tensor(input_ids[:, image_embeddings_length:], mstype.int32)
            text_embeddings = self.to_text_embeddings(text_input_ids)

            concat_inputs_embeds, concat_inputs_attention_mask = self.image_text_concat(image_embeddings,
                                                                                        text_embeddings,
                                                                                        text_input_ids)
            return {
                "input_ids": ms.Tensor(input_ids, mstype.int32),
                "input_embeddings": concat_inputs_embeds,
                "attention_mask": concat_inputs_attention_mask,
                "input_position": input_position
            }
        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "input_embeddings": None,
            "attention_mask": None,
            "input_position": input_position
        }
