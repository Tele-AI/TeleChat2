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
# ============================================================================
"""SAM Model"""
from typing import Tuple

import mindspore as ms
import mindspore.ops as ops

from mindformers.models.build_model import build_network
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from .sam_config import SamConfig


class SamPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SamConfig
    base_model_prefix = "sam"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class SamModel(SamPreTrainedModel):
    """
    SAM predicts object masks from an image and input prompts.
    If prompts are not known in advance, using SamModel is recommended over calling the model directly.
    """
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    _support_list = MindFormerBook.get_model_support_list()['sam']

    def __init__(self, config) -> None:
        super().__init__(config)
        self.image_encoder = build_network(config.image_encoder)
        self.prompt_encoder = build_network(config.prompt_config)
        self.mask_decoder = build_network(config.decoder_config)

        self.load_checkpoint(config)

    def construct(self,
                  image=None,
                  features=None,
                  input_size=None,
                  original_size=None,
                  point_coords=None,
                  point_labels=None,
                  boxes=None,
                  mask_inputs=None,
                  multimask_output=True,
                  return_logits=False):
        """
        Args:
            'image': The image as a mindspore tensor in 3xHxW format,
            already transformed for input to the model.
            'features': The features of an image, extracted by image encoder.
            'input_size': (tuple(int, int)) The input size of
            the image after transformation, as (H, W).
            'original_size': (tuple(int, int)) The original size of
            the image before transformation, as (H, W).
            'point_coords': (ms.Tensor) Batched point prompts for
            this image, with shape BxNx2. Already transformed to the
            input frame of the model.
            'point_labels': (ms.Tensor) Batched labels for point prompts,
            with shape BxN.
            'boxes': (ms.Tensor) Batched box inputs, with shape Bx4.
            Already transformed to the input frame of the model.
            'mask_inputs': (ms.Tensor) Batched mask inputs to the model,
            in the form Bx1xHxW.
            multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
            return_logits (bool): Whether the mask return logits.

        Returns:
            'masks': (ms.Tensor) Batched binary mask predictions,
            with shape BxCxHxW, where B is the number of input prompts,
            C is determined by multimask_output, and (H, W) is the
            original size of the image.
            'iou_predictions': (ms.Tensor) The model's predictions
            of mask quality, in shape BxC.
            'low_res_logits': (ms.Tensor) Low resolution logits with
            shape BxCxHxW, where H=W=256. Can be passed as mask input
            to subsequent iterations of prediction.
        """
        if image is None and features is None:
            raise ValueError("image and feature can't be both None.")

        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point labels can't be None, when point coords is not None.")

        if features is None:
            features = self.image_encoder(image)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(point_coords=point_coords,
                                                                  point_labels=point_labels,
                                                                  boxes=boxes,
                                                                  mask_inputs=mask_inputs)

        low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=features,
                                                           image_pe=self.prompt_encoder.get_dense_pe(),
                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                           dense_prompt_embeddings=dense_embeddings,
                                                           multimask_output=multimask_output)

        masks = self.postprocess_masks(low_res_masks,
                                       input_size=input_size,
                                       original_size=original_size)

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, low_res_masks, iou_predictions

    def postprocess_masks(self,
                          masks: ms.Tensor,
                          input_size: ms.Tensor,
                          original_size: Tuple[int, ...]) -> ms.Tensor:
        """postprocess masks"""
        masks = ops.interpolate(
            masks,
            size=(self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=True
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = ops.interpolate(masks, size=original_size, mode="bilinear", align_corners=True)
        return masks
