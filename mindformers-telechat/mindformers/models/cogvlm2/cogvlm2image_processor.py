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
CogVLM2Image Processor
"""
from typing import Optional

import numpy as np
import mindspore as ms
from PIL import Image

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.transforms.vision_transforms import (
    BatchToTensor,
    BatchNormalize,
)
from mindformers.models.image_processing_utils import BaseImageProcessor

from ..multi_modal import ModalContentTransformTemplate, BaseTextContentBuilder
from ..multi_modal.base_multi_modal_processor import BatchResizeV2
from ..multi_modal.modal_content import ModalContentBuilder
from ..multi_modal.utils import DataRecord


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CogVLM2ImageImageProcessor(BaseImageProcessor):
    """
    CogVLM2ForImage ImageProcessor.

    Args:
        image_size (int): The target size.
        interpolation (str): interpolate method, default is 'cubic'.
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        is_hwc (bool, optional): Whether the input image is HWC.
            ``True`` - HWC format, ``False`` - CHW format. Default: ``True``.
    """

    def __init__(
            self,
            image_size: Optional[int] = 1344,
            interpolation: Optional[str] = "bicubic",
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            is_hwc=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.resize = BatchResizeV2([image_size, image_size], interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        self.normalize = BatchNormalize(mean, std, is_hwc)

    def preprocess(self, images, **kwargs) -> ms.Tensor:
        images = self.resize(images)
        images = self.to_tensor(images)
        images = self.normalize(images)
        return images


# pylint: disable=W0223
class CogVLM2ImageImageBuilder(ModalContentBuilder):
    """
    CogVLM2ForImage Content Builder.
    """

    def __init__(
            self,
            context_pad_token="<|reserved_special_token_2|>",
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            pad_length=None,
            image_size=1344,
            patch_size=14,
    ):
        vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2
        super(CogVLM2ImageImageBuilder, self).__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=vision_token_num,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
        )
        self.vision_token_type = 1
        self.language_token_type = 0
        self.pad_length = pad_length
        self.image_processor = CogVLM2ImageImageProcessor(image_size)

    def regular_input_for_predict(
            self, inputs, result_recorder: DataRecord = None, **kwargs
    ):
        image_path = inputs["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(image)
        result_recorder.put("images", image)

        text = self.start_token + self.end_token
        return {"image": text}

    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        start_position = np.where(input_ids == self.start_token_id)[0]

        if start_position.size == 0:
            return input_ids
        if start_position.size > 1:
            raise ValueError("Multiple image input is not supported.")

        input_ids = np.insert(
            input_ids,
            start_position[0],
            [self.context_pad_token_id] * self.context_length,
        )
        input_ids = np.insert(input_ids, [0], self.tokenizer.bos_token_id)
        input_ids = np.delete(input_ids, np.where(input_ids == self.start_token_id))
        input_ids = np.delete(input_ids, np.where(input_ids == self.end_token_id))

        return input_ids

    def generate_context_positions(
            self, input_ids, batch_index=0, result_recorder: DataRecord = None, **kwargs
    ):
        if self.pad_length is not None:
            input_len = len(input_ids)
            if input_len > self.pad_length:
                raise ValueError("length of input_ids is longger than pad_length.")
            input_ids = np.pad(input_ids,
                               (0, self.pad_length - input_len),
                               "constant",
                               constant_values=self.tokenizer.pad_token_id)
        token_type_ids = (input_ids == self.context_pad_token_id).astype(np.float32)
        vision_token_mask, language_token_mask = self._get_expert_mask(token_type_ids)
        if vision_token_mask is not None:
            vision_indices = self._generate_context_positions(vision_token_mask, True, batch_index)
            image_context_pos = self._generate_context_positions(token_type_ids, self.vision_token_type, batch_index)
        else:
            vision_indices = None
            image_context_pos = None
        language_indices = self._generate_context_positions(language_token_mask, True, batch_index)
        position_ids = self._build_position_ids(token_type_ids)
        result_recorder.put("vision_token_mask", vision_token_mask)
        result_recorder.put("language_token_mask", language_token_mask)
        result_recorder.put("vision_indices", vision_indices)
        result_recorder.put("language_indices", language_indices)
        result_recorder.put("position_ids", position_ids)
        return image_context_pos

    def _generate_context_positions(self, token_mask, target_token_id, batch_index=0):
        context_length = np.sum(token_mask.astype(np.int32))
        pos = np.where(np.array(token_mask) == target_token_id)[0]
        pos = np.expand_dims(pos, axis=0)
        pos = np.insert(pos, 0, batch_index, axis=0)
        pos = np.transpose(pos).reshape((-1, context_length, 2))
        return pos

    def _get_expert_mask(self, token_type_ids):
        vision_token_mask = np.zeros_like(token_type_ids).astype(np.bool_)
        vision_token_mask[:-1] = (token_type_ids[:-1] == self.vision_token_type) & (
            token_type_ids[1:] == self.vision_token_type
        )
        language_token_mask = ~vision_token_mask
        if not vision_token_mask.any():
            vision_token_mask = None
        return vision_token_mask, language_token_mask

    def _build_position_ids(self, x):
        """build position_ids"""
        tmp = x.copy()
        # image boi eoi token as LANGUAGE_TOKEN_TYPE
        is_boi_eoi = np.zeros_like(x).astype(np.bool_)
        is_boi_eoi[1:] |= (tmp[1:] == self.vision_token_type) & (
            tmp[:-1] == self.language_token_type
        )
        is_boi_eoi[0] |= tmp[0] == self.vision_token_type
        is_boi_eoi[:-1] |= (tmp[:-1] == self.vision_token_type) & (
            tmp[1:] == self.language_token_type
        )
        is_boi_eoi[-1] |= tmp[-1] == self.vision_token_type
        tmp[is_boi_eoi] = self.language_token_type
        # final position ids
        y = np.zeros_like(x)
        y[1:] = (tmp[1:] == self.language_token_type) | (
            (tmp[1:] == self.vision_token_type) & (tmp[:-1] == self.language_token_type)
        )
        y = y.cumsum(axis=-1)
        return y


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class CogVLM2ImageContentTransformTemplate(ModalContentTransformTemplate):
    """
    CogVLM2ForImage Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(
            self,
            output_columns=None,
            pad_length=None,
            vstack_columns=None,
            tokenizer=None,
            mode="predict",
            signal_type="chat",
            *args,
            **kwargs,
    ):
        if output_columns is None:
            output_columns = [
                "input_ids",
                "images",
                "image_context_pos",
                "position_ids",
                "vision_token_mask",
                "language_token_mask",
                "vision_indices",
                "language_indices",
            ]
        if vstack_columns is None:
            vstack_columns = [
                "image_context_pos",
                "vision_indices",
                "language_indices",
            ]
        super().__init__(
            output_columns=output_columns,
            vstack_columns=vstack_columns,
            tokenizer=tokenizer,
            mode=mode,
            *args,
            **kwargs,
        )
        self.signal_type = signal_type

        self.modal_builders = {
            "image": CogVLM2ImageImageBuilder(
                use_custom_token=False,
                start_token="<|reserved_special_token_3|>",
                end_token="<|reserved_special_token_4|>",
                pad_length=pad_length,
            ),
            "text": BaseTextContentBuilder(),
        }

    def _history_to_prompt(self, signal_type, history, query):
        """apply prompt template"""
        if signal_type == "base":
            return query
        if signal_type == "vqa":
            answer_format = "Short answer:"
        elif signal_type == "chat":
            answer_format = "Answer:"
        else:
            raise TypeError(f"Unknown signal type {signal_type}")

        prompt = ""
        for old_query, response in history:
            prompt += (
                "Question: "
                + old_query
                + " {} ".format(answer_format)
                + response
                + "\n"
            )
        prompt += "Question: {} {}".format(query, answer_format)
        return prompt

    # pylint: disable=W0613
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        """apply conversion prompt"""
        images = []
        texts = []
        for raw in raw_inputs:
            if isinstance(raw, dict):
                images.append(raw)
            else:
                texts.append(raw)
        if len(images) > 1:
            raise ValueError("not support multi images by now.")
        if len(images) == 1:
            image = images[0]["image"]
        else:
            image = ""
        query = "".join(texts)
        query = self._history_to_prompt(self.signal_type, self.history, query)
        return image + query
