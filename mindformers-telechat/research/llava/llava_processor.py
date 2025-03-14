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
LLaVaProcessor
"""
from typing import Dict
import numpy as np

import mindspore as ms

from mindformers import CLIPImageProcessor
from mindformers import ModalContentTransformTemplate
from mindformers.models.multi_modal.modal_content import ModalContentBuilder
from mindformers.models.multi_modal.utils import DataRecord
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


class LlavaTextBuilder(ModalContentBuilder):
    """Llava Text Content Builder."""

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token=False,
            start_token=None,
            end_token=None,
            tokenizer=None,
            need_create_context_pos=False
    ):
        super(LlavaTextBuilder, self).__init__(
            type_="text",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
            need_create_context_pos=need_create_context_pos
        )

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        if "text" in inputs:
            return inputs["text"]
        return ""

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        return input_ids


class LlavaImageBuilder(ModalContentBuilder):
    """Llava Image Content Builder."""

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            image_size=336
    ):
        super(LlavaImageBuilder, self).__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer
        )
        self.image_processor = CLIPImageProcessor(image_size)

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        image_path = inputs.get("image")
        images = load_image(image_path)
        images = self.image_processor(images)
        self.num_frames = images.shape[0]
        result_recorder.put("images", images)
        return f"{self.context_pad_token}\n"

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        start_position = np.where(input_ids == self.start_token_id)[0]
        offset = 0
        for start_position_item in start_position:
            start_position_item = start_position_item + offset

            input_ids = np.insert(input_ids, start_position_item,
                                  [self.context_pad_token_id] * (self.context_length - 1))
            offset += self.context_length

        return input_ids


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class LlavaContentTransformTemplate(ModalContentTransformTemplate):
    """
    Llava Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(self, output_columns, tokenizer, mode="predict", signal_type="base", *args, **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode, *args, **kwargs)
        self.signal_type = signal_type
        self.context_length = kwargs.get("context_length", 576)
        self.image_size = kwargs.get("image_size", 336)
        self.modal_builders = {
            "image": LlavaImageBuilder("<image>", self.context_length, use_custom_token=False,
                                       start_token="<image>",
                                       end_token="<image>", image_size=self.image_size),

            "text": LlavaTextBuilder("<image>", self.context_length, use_custom_token=False,
                                     start_token="<image>",
                                     end_token="<image>", need_create_context_pos=False),
        }
        self.template = """USER: {image_tags}{query} ASSISTANT:"""
        self.text_instruct_mode = kwargs.get("text_instruct_mode", False)

    # pylint: disable=C0111
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        if len(raw_inputs) == 2:
            prompt = self.template.format(image_tags=raw_inputs[0], query=raw_inputs[1])
            result_recorder.put("no_image_tag", [False])
        else:
            if self.text_instruct_mode:
                prompt = self.template.format(image_tags="", query=raw_inputs[0])
            else:
                prompt = raw_inputs[0]
            prompt += self.tokenizer.image_token
            result_recorder.put("no_image_tag", [True])

        return prompt

    # pylint: disable=C0111
    def batch(self, data_list, token_padding_length, **kwargs):
        batched_data = super().batch(data_list, token_padding_length, **kwargs)

        batch_no_image = batched_data.pop("no_image_tag")
        batch_input_ids = batched_data.get("input_ids")
        image_indices = batched_data.get("image_context_pos")
        batch_images = batched_data.get("images", None)

        if batch_no_image.sum():
            if isinstance(image_indices, ms.Tensor):
                image_indices = image_indices.asnumpy()
            batch_indices = image_indices[batch_no_image.reshape(-1,)].reshape(-1, 2)[:, 0]
            batch_col_indices = image_indices[batch_no_image.reshape(-1,)].reshape(-1, 2)[:, 1]
            batch_input_ids[batch_indices, batch_col_indices] = self.tokenizer.pad_token_id

            final_batch_images = ms.Tensor(np.ones((batch_input_ids.shape[0], 3, 336, 336), dtype=np.float32))
            if batch_images is not None:
                final_batch_images[ms.Tensor(~batch_no_image.reshape(-1,))] = batch_images
        else:
            final_batch_images = batch_images

        batched_data["input_ids"] = batch_input_ids
        batched_data["images"] = final_batch_images
        return batched_data

    # pylint: disable=W0613
    def post_process(self, output_ids, **kwargs):
        output = []
        for output_ids_item in output_ids:
            decoded = self.tokenizer.decode(output_ids_item, skip_special_tokens=True)
            output.append(decoded)
        return output
