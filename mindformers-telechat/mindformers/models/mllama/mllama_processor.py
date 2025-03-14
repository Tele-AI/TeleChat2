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
MllamaProcessor
"""
import copy
from typing import Dict, List

import numpy as np
from mindformers.models.multi_modal.modal_content import ModalContentTransformTemplate
from mindformers.models.multi_modal.modal_content import ModalContentBuilder
from mindformers.models.multi_modal.utils import DataRecord
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger

from .image_processing_mllama import MllamaImageProcessor


class MllamaTextBuilder(ModalContentBuilder):
    """
    Mllama Tetxt Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token=False,
            start_token=None,
            end_token=None,
            tokenizer=None,
            pad_token_id=None,
            need_padding_context=True,
            need_create_context_pos=False
    ):
        super(MllamaTextBuilder, self).__init__(
            type_="text",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
            need_padding_context=need_padding_context,
            need_create_context_pos=need_create_context_pos
        )
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        return inputs["text"]

    def regular_input_for_train(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        inputs = inputs.replace(self.start_token, "")
        return inputs


class MllamaImageBuilder(ModalContentBuilder):
    """
    Mllama Image Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            context_length,
            max_num_images,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            image_token=None,
            tokenizer=None,
            image_size=None,
            need_create_context_pos=False,
            image_mean=None,
            image_std=None
    ):
        super(MllamaImageBuilder, self).__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
            need_create_context_pos=need_create_context_pos
        )
        self.image_token = image_token if image_token is not None else "<|image|>"
        self.image_size = image_size if image_size is not None else 560
        self.max_num_images = max_num_images
        self.image_processor = MllamaImageProcessor(size=self.image_size, image_mean=image_mean,
                                                    image_std=image_std,
                                                    max_num_images=self.max_num_images)

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        text, image_paths = self.find_tag_and_extract(inputs, self.start_token, self.end_token)
        text = text.replace(self.start_token, "").replace(self.end_token, "")
        images = []
        if len(images) > self.max_num_images:
            raise ValueError(f"Invalid input image num, the num should be less than the max_num_images. "
                             f"the input is {len(images)}, but the max_num_images is {self.max_num_images}")
        for image_path in image_paths:
            image = load_image(image_path)
            images.append(image)

        pixel_values, aspect_ratio_ids, aspect_ratio_mask, num_tiles = self.image_processor(images)
        result_recorder.put("no_image_tag", [False])
        result_recorder.put("pixel_values", pixel_values[0])
        result_recorder.put("aspect_ratio_ids", aspect_ratio_ids[0])
        result_recorder.put("aspect_ratio_mask", aspect_ratio_mask[0])
        result_recorder.put("num_tiles", num_tiles[0])

        return text

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        if isinstance(inputs['image'], (list, tuple)):
            image_paths = inputs['image']
        elif isinstance(inputs['image'], str):
            image_paths = [inputs['image']]
        else:
            raise ValueError("Invalid input type. Must be a string image path, a list of images path.")
        images = []
        if len(images) > self.max_num_images:
            raise ValueError(f"Invalid input image num, the num should be less than the max_num_images. "
                             f"the input is {len(images)}, but the max_num_images is {self.max_num_images}")
        for image_path in image_paths:
            image = load_image(image_path)
            images.append(image)

        if images:
            pixel_values, aspect_ratio_ids, aspect_ratio_mask, num_tiles = self.image_processor(images)
            result_recorder.put("no_image_tag", [False])
            result_recorder.put("pixel_values", pixel_values[0])
            result_recorder.put("aspect_ratio_ids", aspect_ratio_ids[0])
            result_recorder.put("aspect_ratio_mask", aspect_ratio_mask[0])
            result_recorder.put("num_tiles", num_tiles[0])
        else:
            result_recorder.put("no_image_tag", [True])

        return "<|image|>"

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        return input_ids


def get_cross_attention_token_mask(input_ids: List[int], image_token_id: int) -> List[List[int]]:
    """
    Generate a cross-attention token mask for image tokens in the input sequence.
    This function identifies the positions of image tokens in the input sequence and creates
    a mask that defines which subsequent tokens each image token should attend to.
    """
    image_token_locations = [i for i, token in enumerate(input_ids) if token == image_token_id]

    if not image_token_locations:
        return []

    # only one image present, unmask until end of sequence
    if len(image_token_locations) == 1:
        return [[image_token_locations[0], -1]]

    vision_masks = [[loc1, loc2] for loc1, loc2 in zip(image_token_locations[:-1], image_token_locations[1:])]

    # last image will attend to all subsequent text
    vision_masks.append([image_token_locations[-1], len(input_ids)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]

    return vision_masks


def convert_sparse_cross_attention_mask_to_dense(
        cross_attention_token_mask: List[List[List[int]]],
        num_tiles: List[List[int]],
        max_num_tiles: int,
        length: int,
) -> np.ndarray:
    """
    Convert the cross attention mask indices to a cross attention mask 4D array.
    """

    batch_size = len(cross_attention_token_mask)
    max_num_images = max([len(masks) for masks in cross_attention_token_mask])
    cross_attention_mask = np.zeros(
        shape=(batch_size, length, max_num_images, max_num_tiles),
        dtype=np.int32,
    )

    for sample_idx, (sample_masks, sample_num_tiles) in enumerate(zip(cross_attention_token_mask, num_tiles)):
        for mask_idx, (locations, mask_num_tiles) in enumerate(zip(sample_masks, sample_num_tiles)):
            if len(locations) == 2:
                start, end = locations
                end = min(end, length)
                if end == -1:
                    end = length
                cross_attention_mask[sample_idx, start:end, mask_idx, :mask_num_tiles] = 1
    return cross_attention_mask


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class MllamaProcessor(ModalContentTransformTemplate):
    """
    Mllama Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(self, output_columns, tokenizer, mode="predict", signal_type="base", *args, **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode, *args, **kwargs)
        self.signal_type = signal_type
        self.context_length = kwargs.get("context_length", 576)
        self.image_size = kwargs.get("image_size", 560)
        self.ignore_token_id = kwargs.get("ignore_token_id", -100)
        self.pad_token_id = kwargs.get("pad_token_id", 128004)
        self.image_token_id = kwargs.get("image_token_id", 128256)
        self.image_token = "<|image|>"
        self.add_special_tokens = kwargs.get("add_special_tokens", True)
        self.max_image_tiles = kwargs.get("max_image_tiles", 4)
        self.image_mean = kwargs.get("image_mean", None)
        self.image_std = kwargs.get("image_std", None)
        self.max_num_images = kwargs.get("max_num_images", None)
        self.need_padding_context = False
        if self.mode == "predict":
            self.need_padding_context = True
        self.modal_builders = {
            "image": MllamaImageBuilder("<|image|>", self.context_length, use_custom_token=False,
                                        need_create_context_pos=False,
                                        start_token='<|reserved_special_token_3|>',
                                        end_token='<|reserved_special_token_4|>', image_size=self.image_size,
                                        image_mean=self.image_mean, image_std=self.image_std,
                                        max_num_images=self.max_num_images),

            "text": MllamaTextBuilder("<|text|>", self.context_length, use_custom_token=False,
                                      start_token="<|text|>", end_token="<|text|>", pad_token_id=self.pad_token_id,
                                      need_padding_context=self.need_padding_context, need_create_context_pos=False),
        }
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|eot_id|>"
        system_prompt = ["<|start_header_id|>", b"system", "<|end_header_id|>"]
        user_prompt = ["<|start_header_id|>", b"user", "<|end_header_id|>"]
        assistant_prompt = ["<|start_header_id|>", b"assistant", "<|end_header_id|>"]

        self.system_prompt_id = self.tokenizer.convert_tokens_to_ids(system_prompt)
        self.user_prompt_id = self.tokenizer.convert_tokens_to_ids(user_prompt)
        self.assistant_prompt_id = self.tokenizer.convert_tokens_to_ids(assistant_prompt)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

    def build_conversation_input_text(self, raw_inputs, result_recorder):
        """build conversation input of text"""

        if self.mode == 'train':
            return self.build_train_conversation(raw_inputs, result_recorder)
        if self.mode == 'predict':
            return self.build_predict_conversation(raw_inputs, result_recorder)

        raise ValueError(f"Wrong run mode! Current run mode is {self.mode},  please select in "
                         f" ['train', 'predict']")

    def build_train_conversation(self, raw_inputs, result_recorder):
        """build train conversation"""
        # Apply prompt templates
        conversations = []
        # Skip the first one if it is not from human

        conversation = self.get_prompt(raw_inputs)

        conversations.append(conversation)
        result_recorder.put("conversations", conversations)
        return conversations

    def build_labels(self, text_id_list, result_recorder, **kwargs):

        label_list = []
        for text_id in text_id_list:
            dialog_tokens = text_id.tolist()
            labels = copy.copy(dialog_tokens)
            eot_indices = [j for j, n in enumerate(labels) if n == self.eos_token_id]
            last_idx = 0
            # system prompt header "<|start_header_id|>system<|end_header_id|>"
            # user prompt header "<|start_header_id|>user<|end_header_id|>"
            prompt_header_seqs = [self.system_prompt_id, self.user_prompt_id]
            for _, idx in enumerate(eot_indices):
                current_seq = labels[last_idx:idx + 1]
                if self.check_header(prompt_header_seqs, current_seq):
                    # found prompt header, indicating that this seq should be masked
                    labels[last_idx:idx + 1] = [self.ignore_token_id] * (idx - last_idx + 1)
                else:
                    last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>
            assistant_header_seq = self.assistant_prompt_id
            labels = self.replace_target(assistant_header_seq, labels)
            # Mask the padding token and image token
            for pos, label in enumerate(labels):
                if label in [self.pad_token_id, self.image_token_id]:
                    labels[pos] = self.ignore_token_id
            label_list.append(labels)

        return label_list[0]

    def get_need_update_output_items(self, result: DataRecord):
        """
        Retrieve the output items that need to be updated.

        Args:
            result (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            A Dict. Defaults to an empty dict.
        """
        no_image_tag = result.get("no_image_tag")[0]
        input_ids = result.get("input_ids")
        if not no_image_tag:
            num_tiles = result.get("num_tiles")
            cross_attention_token_mask = [get_cross_attention_token_mask(input_ids, self.image_token_id)]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=[num_tiles],
                max_num_tiles=self.max_image_tiles,
                length=len(input_ids),
            )
            result.put("cross_attention_mask", cross_attention_mask[0])
        return {}

    def get_prompt(self, conversations):
        prompt = self.bos_token
        for _, (key_from, value) in enumerate(conversations):
            prompt += "<|start_header_id|>" + key_from + "<|end_header_id|>\n\n" + value + self.eos_token
        return prompt

    def build_string_from_input(self, prompt, bos_token, image_token):
        """
        Builds a string from the input prompt by adding `bos_token` if not already present.

        Args:
            prompt (`str`):
                The input prompt string.
            bos_token (`str`):
                The beginning of sentence token to be added.
            image_token (`str`):
                The image token used to identify the start of an image sequence.

        Returns:
            str: The modified prompt string with the `bos_token` added if necessary.

        Examples:
            >>> build_string_from_input("Hello world", "<begin_of_text>", "<|image|>")
            '<begin_of_text>Hello world'

            >>> build_string_from_input("<|image|>Hello world", "<begin_of_text>", "<|image|>")
            '<|image|><begin_of_text>Hello world'

            >>> build_string_from_input("<begin_of_text>Hello world", "<begin_of_text>", "<|image|>")
            '<begin_of_text>Hello world'
        """

        if bos_token in prompt:
            return prompt

        num_image_tokens_on_start = 0
        while prompt.startswith(image_token):
            prompt = prompt[len(image_token):]
            num_image_tokens_on_start += 1

        return f"{image_token * num_image_tokens_on_start}{bos_token}{prompt}"

    def build_predict_conversation(self, raw_inputs, result_recorder):
        prompt = self.build_string_from_input(raw_inputs[1], self.bos_token, self.image_token)
        result_recorder.put("prompt", prompt)
        return prompt

    def check_header(self, targets, seq):
        for i in range(len(seq) - 3):
            if seq[i:i + 3] in targets:
                return True
        return False

    def replace_target(self, target, seq):
        for i in range(len(seq) - 3):
            if seq[i:i + 3] == target:
                seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
        return seq

    def batch_input_ids(self, input_ids_list, max_length):
        padded_input_ids = []
        max_length = max([len(i) for i in input_ids_list])
        for input_ids in input_ids_list:
            input_ids = np.pad(input_ids, (0, max_length - len(input_ids)), "constant",
                               constant_values=self.tokenizer.pad_token_id)
            padded_input_ids.append(input_ids)
        return np.stack(padded_input_ids, axis=0)

    def batch(self, data_list, token_padding_length, **kwargs):
        batched_data = {}
        for column_name in self.output_columns:
            column_data_list = [data[column_name] for data in data_list if
                                data[column_name] is not None and len(data[column_name]) > 0]

            if column_name == "input_ids":
                batched_data[column_name] = self.batch_input_ids(column_data_list, token_padding_length)
                continue

            if column_data_list:
                batch_success, batched_column_data = super().try_to_batch(column_data_list, column_name)
                batched_data[column_name] = batched_column_data
                if not batch_success:
                    logger.warning("batching %s failed, try to create a separate process for this data.", column_name)
            else:
                batched_data[column_name] = None
                logger.warning(f"the data {column_name} is empty when batching data, please check the query data.")
        return batched_data
