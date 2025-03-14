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
LLaVaNextProcessor
"""
import collections
from collections import namedtuple
import math
import re
from typing import Dict

import cv2
import mindspore as ms
import numpy as np
from PIL import Image

from mindformers import ModalContentTransformTemplate
from mindformers.models.build_processor import build_processor
from mindformers.models.multi_modal.modal_content import ModalContentBuilder
from mindformers.models.multi_modal.utils import DataRecord
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.utils.image_utils import select_best_resolution
from research.llava_next.llava_anyres_process import LLavaAnyRes

VIDEO_FORMAT_LIST = (
    '3GP', 'wmv', 'asf', 'dot', 'mpg', 'mpeg', 'mov', 'mp4', 'avi', 'flv', 'mpeg', 'f4v', 'mkv', "webm")
IMAGE_FORMAT_LIST = ("bmp", "jpeg", "jpg", "png", "tif", "webp", "apng")
VideoFrames = namedtuple('VideoFrames', ['frames', 'video_time', 'frame_time', 'num_frames_to_sample'])
PaddedImageInfo = collections.namedtuple("PaddedImageInfo", ["images", "image_patches", "patches_unpad_index",
                                                             "total_paded_length"])


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (ms.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, "
                f"np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def expand2square(pil_img, background_color):
    """
    Make the shape of image into square shape

    Args:
        pil_img (`RGB`):
            RGB images
        background_color (`tuple`):
            A list containing image mean number to fill the possible pixel
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def process_video_with_cv2(video_file, data_args):
    """
     Read Videos and change it to a number of arrays

    Args:
        video_file (`str`):
            Video file path
        data_args (`dict`):
            A dict of params for processing video

    Returns:
        tuple: video arrays, video_time, frame_time, num_frames
    """

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = np.linspace(0, frame_num - 1, data_args["frames_upbound"], dtype=np.int32)
    frame_time = [i / fps for i in frame_idx]
    video_time = frame_num / fps

    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    num_frames_to_sample = len(frame_idx)

    frames = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"frames in video may corrupt, please check video file {video_file}.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return VideoFrames(np.array(frames), video_time, frame_time_str, num_frames_to_sample)


class LlavaTextBuilder(ModalContentBuilder):
    """
    Llava Tetxt Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            use_custom_token=False,
            start_token=None,
            end_token=None,
            tokenizer=None,
            need_create_context_pos=False
    ):
        super(LlavaTextBuilder, self).__init__(
            type_="text",
            context_pad_token=context_pad_token,
            context_length=None,
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
    def regular_input_for_train(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        return inputs.replace(self.context_pad_token, "")

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        return input_ids


class LlavaImageAndVideoBuilder(ModalContentBuilder):
    """
    Llava Image and video Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            context_length=729,
            tokenizer=None,
            **kwargs
    ):
        super(LlavaImageAndVideoBuilder, self).__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer
        )
        self.patch_size = kwargs.get("patch_size")
        self.image_processor = build_processor(kwargs.get("image_processor"))
        self.image_size = self.image_processor.size.get("shortest_edge") if \
            self.image_processor.size.get("shortest_edge", None) is not None \
            else min(self.image_processor.size["height"], self.image_processor.size["width"])
        self.anyres_process = LLavaAnyRes(self.image_processor)
        self.vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy")
        self.add_length_offset = kwargs.get("add_length_offset", False)
        self.video_contains_pooler = kwargs.get("video_contains_pooler", False)
        self.one_vision_contains_interpolate = kwargs.get("one_vision_contains_interpolate", False)
        self.add_time_instruction = kwargs.get("add_time_instruction", False)
        self.args = kwargs

    def process_image_input(self, video_paths, result_recorder):
        """
        process image into tensors
        """
        images = []
        image_sizes = []
        if video_paths[0].strip():
            if len(video_paths) > 1:
                result_recorder.put("image_type", "multi_image")
            else:
                result_recorder.put("image_type", "single_image")
            for video_path in video_paths:
                if video_path.split(".")[-1] in VIDEO_FORMAT_LIST:
                    result_recorder.put("image_type", "video")  # overwrite image_type to video
                    videos, video_time, frame_time, num_frames_to_sample = process_video_with_cv2(video_path, self.args)
                    image_sizes.append(videos.shape[-3:-1])
                    images = videos
                    if self.add_time_instruction:
                        time_instruction = f"The video lasts for {video_time:.2f} seconds, " \
                                           f"and {num_frames_to_sample} frames are uniformly sampled from it. " \
                                           f"These frames are located at {frame_time}.Please answer the " \
                                           f"following questions related to this video."
                        result_recorder.put("time_instruction", time_instruction)
                else:
                    videos = load_image(video_path)
                    image_sizes.append(videos.size)
                    images.append(videos)
            do_spatial_unpad = result_recorder.get("image_type") == "single_image"
            result_recorder.put("do_spatial_unpad", do_spatial_unpad)
            if result_recorder.get("image_type") == "multi_image":
                # do multi image expand
                for idx, image in enumerate(images):
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    images[idx] = image
            if do_spatial_unpad:
                # do single image patched
                images = self.anyres_process(images)
            images_dict = self.image_processor(images)
            result_recorder.put("no_image_tag", [False])
            images = images_dict.get("pixel_values")
            result_recorder.put("image_sizes", image_sizes)
        else:
            # no image tag
            images = ms.ops.zeros((1, 3, self.image_size, self.image_size), ms.float32)
            result_recorder.put("no_image_tag", [True])
        result_recorder.put("images", images)
        result_recorder.put("split_size", images.shape[0])

    # pylint: disable=W0613
    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        """process single image, multi image and video """
        text, video_paths = self.find_tag_and_extract(inputs, self.start_token, self.end_token)
        text = text.replace(self.start_token, "").replace(self.end_token, "")
        text = self.change_image_token_pos(text)
        self.process_image_input(video_paths, result_recorder)
        return text

    def change_image_token_pos(self, text):
        """
        Based on https://github.com/LLaVA-VL/LLaVA-NeXT.git
        <image> token will always be put in the start of user's question
        """
        num_im = len(re.findall(self.context_pad_token, text))
        if num_im == 1 and self.context_pad_token in text and not text.startswith(
                self.context_pad_token):
            text = text.replace(self.context_pad_token, "").strip()
            text = self.context_pad_token + "\n" + text
            return text.strip()
        return text

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        """process single image, multi image and video """
        video_paths = inputs.get("image") if "image" in inputs else inputs.get("video")
        if not isinstance(video_paths, list):
            video_paths = [video_paths]
        self.process_image_input(video_paths, result_recorder)
        time_instruction = ""
        if self.add_time_instruction:
            time_instruction = result_recorder.get("time_instruction")
        return f"{self.context_pad_token}{time_instruction}\n"

    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""

        start_position = np.where(input_ids == self.context_pad_token_id)[0]
        if not start_position:
            return input_ids

        context_length = self.cal_context_length(result_recorder)
        if self.vision_feature_select_strategy == "default" and result_recorder.get("image_type") != "video":
            context_length -= 1
        elif result_recorder.get("image_type") == "video":
            context_length *= result_recorder.get("images").shape[0]
            if self.add_length_offset:
                context_length += 1
        self.context_length = context_length
        offset = 0
        for start_position_item in start_position:
            start_position_item = start_position_item + offset

            input_ids = np.insert(input_ids, start_position_item,
                                  [self.context_pad_token_id] * (context_length - 1))
            offset += (context_length - 1)
        return input_ids

    def cal_context_length(self, result_recorder):
        """calculate context length"""
        images = result_recorder.get("images")
        do_spatial_unpad = result_recorder.get("do_spatial_unpad")
        no_image_tag = result_recorder.get("no_image_tag")[0]
        if no_image_tag and images is None:
            images = ms.ops.zeros((1, 3, self.image_size, self.image_size), ms.float32)
            result_recorder.put("images", images)
        shapes = images.shape
        height, width = shapes[-2:]
        if do_spatial_unpad:
            # single image context_length calculation
            image_sizes = result_recorder.get("image_sizes")
            orig_width, orig_height = image_sizes[0]
            context_length = self._get_number_of_features(orig_height, orig_width, height, width)
            return context_length

        if not no_image_tag and result_recorder.get("image_type") == "video":
            # video context_length calculation
            patches_height_width = (height // self.patch_size)
            patches_width_width = (width // self.patch_size)
            # patches_height_width = int(math.sqrt(self.context_length))
            pooled_height_width = math.ceil(patches_height_width / 2)
            if self.video_contains_pooler:
                context_length = (pooled_height_width * pooled_height_width)
            else:
                context_length = patches_height_width * patches_width_width
            return context_length
        if no_image_tag:
            context_height = (height // self.patch_size)
            context_width = (width // self.patch_size) * 2 + 1
            context_length = context_height * context_width + 1
            return context_length
        context_length = (height // self.patch_size) * (width // self.patch_size) + 1
        return context_length

    def _get_number_of_features(self, orig_height, orig_width, height, width):
        """calculate context length for single imaged processed by any res method"""
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints
        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = (height // self.patch_size)
        patches_width = (width // self.patch_size)
        unpadded_features, newline_features = self._get_unpadded_features(
            [orig_height, orig_width], patches_height, patches_width, scale_height, scale_width
        )
        base_features = patches_height * patches_width
        if self.vision_feature_select_strategy == "default":
            base_features += 1
        context_length = unpadded_features + newline_features + base_features
        return context_length

    def _get_unpadded_features(self, ori_size, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        height, width = ori_size
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = (height * current_width) // width
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = (width * current_height) // height
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(current_height * current_width / (9 * patches_height ** 2))
        if ratio > 1.1 and self.one_vision_contains_interpolate:
            unpadded_features = int(current_height // ratio) * int(current_width // ratio)
            newline_features = int(current_height // ratio)
        return (unpadded_features, newline_features)


class ModalBuilder:
    """
    Llava Modal Builder Class for using the same builder of image and builder.
    """
    def __init__(self, context_length, tokenizer, **kwargs):
        self.modal_builders = {
            "image": LlavaImageAndVideoBuilder("<image>", use_custom_token=False, context_length=context_length,
                                               start_token='<|reserved_special_token_3|>',
                                               end_token='<|reserved_special_token_4|>', tokenizer=tokenizer, **kwargs),

            "text": LlavaTextBuilder("<text>", use_custom_token=False,
                                     start_token="<text>",
                                     end_token="<text>", tokenizer=tokenizer, need_create_context_pos=False),
        }

    def get(self, model_type):
        if model_type == "video":
            model_type = "image"
        return self.modal_builders.get(model_type)

    def values(self):
        return self.modal_builders.values()


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class LlavaNextContentTransformTemplate(ModalContentTransformTemplate):
    """
    Llava Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(self, output_columns, tokenizer, mode="predict", signal_type="base", *args, **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode, *args, **kwargs)
        self.args = kwargs
        self.signal_type = signal_type
        self.image_size = kwargs.pop("image_size", 336)
        self.ignore_token_id = kwargs.get("ignore_token_id", -100)
        self.add_special_tokens = kwargs.get("add_special_tokens", True)
        self.frames_upbound = self.args.get("frames_upbound", 10)
        context_length = self.args.get("context_length", 729)
        self.patch_size = self.args.get("patch_size", 14)
        self.modal_builders = ModalBuilder(context_length, tokenizer, **kwargs)
        self.text_instruct_mode = kwargs.get("text_instruct_mode", False)
        self.prompt_processor = build_processor(kwargs.get("prompt_processor"), default_args={"tokenizer": tokenizer})
        self.image_grid_pinpoints = self.modal_builders.get("image").image_processor.image_grid_pinpoints
        self.num_queries = kwargs.get("num_queries", 576)
        self.max_patch_height_num = kwargs.get("max_patch_height_num", 5)
        self.max_patch_width_num = kwargs.get("max_patch_width_num", 5)
        self.height = self.width = int(math.sqrt(self.num_queries))
        self.add_newline = kwargs.get("add_newline", False)
        self.img_dynamic_batch = kwargs.get("img_dynamic_batch", False)
        self.text_dynamic_batch = kwargs.get("text_dynamic_batch", False)
        self._supported_modal = ["text", "image", "video"]

    # pylint: disable=C0111
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        if self.mode == 'train':
            return self.build_train_conversation(raw_inputs, result_recorder)
        if self.mode == 'predict':
            return self.build_predict_conversation(raw_inputs, result_recorder)
        raise ValueError(f"Wrong run mode! Current run mode is {self.mode},  please select in "
                         f" ['train', 'predict']")

    def build_predict_conversation(self, raw_inputs, result_recorder):
        """decorate text by predict prompt"""
        prompt = self.prompt_processor.build_predict_prompt(raw_inputs)
        if len(raw_inputs) == 2:
            result_recorder.put("no_image_tag", [False])
        else:
            if not self.text_instruct_mode:
                prompt = raw_inputs[0]
            prompt += self.tokenizer.image_token
            result_recorder.put("no_image_tag", [True])

        return prompt

    def build_train_conversation(self, raw_inputs, result_recorder):
        """decorate text by train prompt"""
        # Apply prompt templates
        conversations = self.prompt_processor.build_prompt(raw_inputs, result_recorder)
        result_recorder.put("conversations", conversations)
        return conversations

    def get_need_update_output_items(self, result: DataRecord):
        """
        Retrieve the output items that need to be updated.
        Purpose:
        video shape: [frames, channel, height, width]
        multi image shape: [num, channel, height, width]
        single image shape: [1, channel, height, width]
        make the first dimension union to the largest frames.
        Args:
            result (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            A Dict. Defaults to an empty dict.
        """
        no_image_tag = result.get("no_image_tag")[0]
        input_ids = result.get("input_ids")
        image_indices = result.get("image_context_pos")
        images = result.get("images").asnumpy()
        image_size = result.get("image_sizes")
        labels = result.get("labels")
        image_type = result.get("image_type")
        if no_image_tag:
            # process no image situaiton in input_ids
            if isinstance(image_indices, ms.Tensor):
                image_indices = image_indices.asnumpy()
            batch_col_indices = image_indices.reshape(-1, 2)[:, 1]
            input_ids[batch_col_indices] = self.tokenizer.pad_token_id
            image_patches = np.zeros((1, 1, 3, self.image_size, self.image_size), np.float32)
            result.put("image_patches", image_patches)
        else:
            if image_type == "single_image":
                paded_image_info = self.image_process(images, image_size)
                # known bug from transformers/models/llava_next_video/modeling_llava_next_video.py
                # for high precision image, the preprocessed expected length will not be equal to the true length
                input_ids, labels = self._update_input_for_high_precision_image(input_ids, labels, paded_image_info)
                image_indices = self.update_image_position(input_ids, paded_image_info.patches_unpad_index,
                                                           paded_image_info.total_paded_length)
                images = paded_image_info.images
                result.put("image_patches", paded_image_info.image_patches)

        if self.text_dynamic_batch:
            padd_index = np.where(input_ids == self.tokenizer.pad_token_id)[0]
            if padd_index:
                diviser = 2
                remainder = 1
                num = padd_index[0] // diviser
                dyn_length = (num + 1) * diviser + remainder
                input_ids = input_ids[:dyn_length]
                labels = labels[:dyn_length]
        if self.mode == "train":
            result.put("image_context_pos", image_indices.reshape(1, -1, 2))
            result.put("images", images)
            if no_image_tag:
                labels[batch_col_indices] = self.ignore_token_id
        else:
            result.put("image_context_pos", ms.Tensor(image_indices))
            result.put("images", ms.Tensor(images))
            if image_type == "single_image":
                result.put("image_patches", ms.Tensor(paded_image_info.image_patches[np.newaxis, :, :, :, :, :]))
            if no_image_tag:
                result.put("image_patches", ms.Tensor(result.get("image_patches")[np.newaxis, :, :, :, :, :]))
        result.put("input_ids", input_ids)
        result.put("labels", labels)
        return {}

    def _update_input_for_high_precision_image(self, input_ids, labels, paded_image_info):
        """
        update image placeholder for very high precision image
        Args:
            input_ids (``np.array``):
                input_ids with image placeholder
            labels (int):
                labels with image placeholder
            paded_image_info (`namedtuple`):
                nametuple with image patches information
        Returns:
            (input_ids, labels)
        """
        pos = np.where(np.array(input_ids) == self.modal_builders.get("image").context_pad_token_id)[0]
        unpad_image_pos = pos[self.num_queries:]
        # see the length of patches_unpad_index as golden length
        golden_length = len(paded_image_info.patches_unpad_index)
        target_length = len(input_ids)
        if len(unpad_image_pos) == golden_length:
            return input_ids, labels
        if len(unpad_image_pos) > golden_length:
            # shrink the length of input ids and labels
            deleted_pos = pos[golden_length + self.num_queries:]
            input_ids = np.delete(input_ids, deleted_pos)
            current_length = len(input_ids)
            padding_length = target_length - current_length
            input_ids = np.pad(input_ids, (0, padding_length), "constant", constant_values=self.tokenizer.pad_token_id)
            if labels is not None:
                labels = np.delete(labels, deleted_pos)
                labels = np.pad(labels, (0, padding_length), "constant", constant_values=-100)
        else:
            # expand the length of input ids and labels
            added_value_for_inputs = [self.tokenizer.img_token_id] * (golden_length - len(unpad_image_pos))
            input_ids = np.insert(input_ids, pos[-1] + 1, added_value_for_inputs)
            input_ids = input_ids[:target_length]
            if labels is not None:
                added_values_for_labels = [-100] * (golden_length - len(unpad_image_pos))
                labels = np.insert(labels, pos[-1] + 1, added_values_for_labels)
                labels = labels[:target_length]
        return input_ids, labels

    def image_process(self, images, image_sizes):
        """get single image padded index, used to get real image position in context length"""
        ori_width, ori_height = image_sizes[0]
        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            (ori_height, ori_width),
            self.image_grid_pinpoints,
            self.image_size,
        )

        image_patched = images[1:]
        _, channel, height, width = images.shape
        image_patched = image_patched.reshape((num_patch_height, num_patch_width, channel, height, width))
        row_num = self.height * num_patch_height
        col_num = self.width * num_patch_width
        patch_weight = height // self.patch_size
        patch_width = width // self.patch_size

        if not self.img_dynamic_batch:
            if self.max_patch_height_num < num_patch_height or self.max_patch_width_num < num_patch_width:
                raise ValueError(f"Max (height, width) num is ({self.max_patch_height_num, self.max_patch_width_num}),"
                                 f" smaller than current patch num ({num_patch_height}, {num_patch_width}), "
                                 f"please increase max num.")
            image_patched_height_paded = self.padding_to_largest(image_patched, self.max_patch_height_num,
                                                                 num_patch_height, dim=0)
            image_patched = self.padding_to_largest(image_patched_height_paded,
                                                    self.max_patch_width_num, num_patch_width, dim=1)
            row_num = self.max_patch_height_num * self.height
            col_num = self.max_patch_width_num * self.width

        if self.add_newline:
            col_num += 1
        patches_unpad_index = self.unpad_image((patch_weight * num_patch_height,
                                                patch_width * num_patch_width), (ori_height, ori_width), col_num)

        total_paded_length = row_num * col_num + self.height * self.width
        return PaddedImageInfo(images[0][np.newaxis, :, :, :], image_patched,
                               patches_unpad_index, total_paded_length)

    def padding_to_largest(self, images, max_num, current_num, dim):
        """
        padding the dim of dynamic patch into max num
        Args:
            images (``np.array``):
                image numpy array
            max_num (int):
                the max num to pad
            current_num (int):
                the current num of the padded dim
            dim (int):
                the axis dim to pad
        Returns:
            `np.array`: The padded patched numpy array
        """
        if max_num == current_num:
            return images
        image_shape = list(images.shape)
        image_shape[dim] = 1
        padding_values = np.zeros(image_shape, dtype=np.float32)
        image_list = [images] + [padding_values] * (max_num - current_num)
        paded_images = np.concatenate(image_list, axis=dim)
        return paded_images

    def unpad_image(self, current_size, original_size, col_num):
        """
        get unpad image tensor index.

        Args:
            current_size (`tuple`)::
                The current size of image.
            original_size (`tuple`):
                The original size of the image (height, width).
            row_num: the total row number in a paded image
            col_num: the total col number in a paded image
        Returns:
            `np.array`: The unpaded image tensor index will be used for image position
        """

        def calculate_row(start, end, origin_column_num=48, pad_column_num=241, add_new_line=True):
            # 48 * 49
            for row_num in range(start, end):
                need_row_index = np.arange(row_num * pad_column_num, origin_column_num + row_num * pad_column_num)
                new_line_index = np.array((pad_column_num - 1) + (row_num * pad_column_num)).reshape((1,))
                if row_num == start:
                    need_index = need_row_index
                    if add_new_line:
                        need_index = np.concatenate([need_row_index, new_line_index])
                else:
                    if add_new_line:
                        need_index = np.concatenate([need_index, need_row_index, new_line_index])
                    else:
                        need_index = np.concatenate([need_index, need_row_index])

            return need_index

        def calculate_col(start, end, row_num=48, column_num=49, add_new_line=True):
            # 48 * 49
            for i in range(row_num):
                need_col_index = np.arange(column_num * i + start, column_num * i + end)
                new_line_index = np.array((column_num - 1) + (column_num * i)).reshape((1,))
                if i == 0:
                    if add_new_line:
                        need_index = np.concatenate((need_col_index, new_line_index))
                    else:
                        need_index = need_col_index
                else:
                    if add_new_line:
                        need_index = np.concatenate((need_index, need_col_index, new_line_index))
                    else:
                        need_index = np.concatenate((need_index, need_col_index))
            return need_index

        if not isinstance(original_size, (list, tuple)):
            if not isinstance(original_size, (ms.Tensor, np.ndarray)):
                raise TypeError(
                    f"image_size invalid type: {type(original_size)} not valid, should be either list, tuple, "
                    f"np.ndarray or tensor"
                )
            original_size = original_size.tolist()
        original_height, original_width = original_size
        current_height, current_width = current_size

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            end = current_height - padding
            chosen_index = calculate_row(padding, end, current_size[1], col_num, self.add_newline)
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            end = current_width - padding
            chosen_index = calculate_col(padding, end, current_size[0], col_num, self.add_newline)

        return chosen_index

    def update_image_position(self, input_ids, unpad_index, total_paded_length):
        """
        get unpad image tensor index.

        Args:
            input_ids (`np.array`)::
                The current input id
            unpad_index (`np.array`):
                The real used patches image location in final position
            total_paded_length(`str`):
                The total context length
        Returns:
            `np.array`: The final image position with used patches image position and the original image
        """
        pos = np.where(np.array(input_ids) == self.modal_builders.get("image").context_pad_token_id)[0]
        base_image_pos = pos[:self.num_queries]
        unpad_image_pos = pos[self.num_queries:]
        final_image_pos = np.ones((total_paded_length,), dtype=np.int32) * base_image_pos[-1]
        final_image_pos[-self.num_queries:] = base_image_pos
        final_image_pos[unpad_index] = unpad_image_pos

        final_image_pos = np.expand_dims(final_image_pos, axis=0)
        final_image_pos = np.insert(final_image_pos, 0, 0, axis=0)
        final_image_pos = np.transpose(final_image_pos).reshape((1, -1, 2))
        return final_image_pos

    # pylint: disable=W0613
    def build_labels(self, text_id_list, result_recorder, **kwargs):
        """build labels"""
        return self.prompt_processor.build_labels(text_id_list, result_recorder, self.ignore_token_id,
                                                  context_length=self.modal_builders.get("image").context_length)

    def batch(self, data_list, token_padding_length, **kwargs):
        """get batch for predict tensor"""
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
        shape_ = final_batch_images.shape
        image_pos_shape = image_indices.shape
        final_batch_images = final_batch_images.reshape(1, *shape_)
        image_indices = image_indices.reshape(1, *image_pos_shape)
        batched_data["images"] = final_batch_images
        batched_data["image_context_pos"] = image_indices
        return batched_data

    # pylint: disable=W0613
    def post_process(self, output_ids, **kwargs):
        output = []
        for output_ids_item in output_ids:
            decoded = self.tokenizer.decode(output_ids_item, skip_special_tokens=True)
            output.append(decoded)
        return output
