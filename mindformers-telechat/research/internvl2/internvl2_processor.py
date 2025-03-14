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
Internvl2 Data Processor
"""
from typing import Union, List, Dict
from PIL import Image
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore.dataset import vision

from mindformers.dataset import Resize
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.multi_modal import ModalContentTransformTemplate
from mindformers.models.multi_modal.modal_content import ModalContentBuilder
from mindformers.models.multi_modal.utils import DataRecord
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from research.internvl2.conversation import get_conv_template

__all__ = ['InternVLImageContentTransformTemplate']

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLTextBuilder(ModalContentBuilder):
    """InternVL Text Content Builder."""

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token=False,
            tokenizer=None,
            need_create_context_pos=False
    ):
        super(InternVLTextBuilder, self).__init__(
            type_="text",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            tokenizer=tokenizer,
            need_create_context_pos=need_create_context_pos
        )

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        _ = result_recorder
        if "text" in inputs:
            return inputs["text"]
        return ""

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        _ = result_recorder
        return input_ids


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class InternVLImageProcessor(BaseImageProcessor):
    """InternVL Image Processor"""
    def __init__(self, input_size=448, max_num=12, use_thumbnail=False, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        """Find the aspect ratio closest to the original proportion"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Get split and thumbnail images"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the target aspect ratios based on min_num and max_num
        target_ratios = set()
        for i in range(1, max_num + 1):
            min_j = max((min_num + i - 1) // i, 1)
            max_j = min(max_num // i, max_num)
            for j in range(min_j, max_j + 1):
                target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height,
                                                             image_size)

        # Calculate target width and height based on aspect ratio
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height), Image.BICUBIC)

        # Split the image into blocks based on the correct logic
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Crop the image using the calculated box
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size), Image.BICUBIC)
            processed_images.append(thumbnail_img)

        return processed_images

    def build_transform(self, input_size):
        """Build the transformation pipeline."""
        means, stds = IMAGENET_MEAN, IMAGENET_STD
        transform = [
            lambda img: img.convert('RGB') if img.mode != 'RGB' else img,
            Resize((input_size, input_size), interpolation='bicubic'),
            vision.ToTensor(),
            vision.Normalize(mean=means, std=stds, is_hwc=False)
        ]
        return transform

    def apply_transform(self, image, transform):
        for t in transform:
            image = t(image)
        return np.array(image)

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.apply_transform(image, transform) for image in images]
        pixel_values = ms.Tensor(np.stack(pixel_values))
        return pixel_values

    # pylint: disable=W0613
    def preprocess(self, images: Union[ms.Tensor, Image.Image, np.ndarray, List[Image.Image]], **kwargs):
        return self.load_image(images)


# pylint: disable=W0223
class InternVLImageBuilder(ModalContentBuilder):
    """InternVL2 Image Builder."""
    def __init__(
            self,
            context_pad_token='<image>',
            context_length=None,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            context_pad_token_id=64000,
    ):
        super(InternVLImageBuilder, self).__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
        )
        self.image_processor = InternVLImageProcessor()
        self.context_pad_token_id = context_pad_token_id

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        image_path = inputs.get("image")
        pixel_values = self.image_processor(image_path)
        self.num_frames = pixel_values.shape[0]
        result_recorder.put("images", pixel_values)
        result_recorder.put('images_shape', pixel_values.shape)
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
class InternVLImageContentTransformTemplate(ModalContentTransformTemplate):
    """
    InternVL2ForImage Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(
            self,
            output_columns=None,
            vstack_columns=None,
            tokenizer=None,
            mode="predict",
            signal_type="chat",
            template="Hermes-2",
            img_start_token='<img>',
            img_end_token='</img>',
            img_context_token='<IMG_CONTEXT>',
            context_pad_token='<image>',
            downsample_ratio=0.5,
            image_size=448,
            patch_size=14,
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
        self.template = template
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        self.img_context_token = img_context_token
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.context_length = num_image_token
        self.context_pad_token = context_pad_token
        self.cast = P.Cast()
        self.signal_type = signal_type
        self.modal_builders = {
            "image": InternVLImageBuilder(
                use_custom_token=False,
                start_token="<|reserved_special_token_3|>",
                end_token="<|reserved_special_token_4|>",
                context_length=num_image_token,
                context_pad_token=context_pad_token,
                context_pad_token_id=kwargs.get("context_pad_token_id")
            ),
            "text": InternVLTextBuilder(
                context_pad_token="<image>",
                context_length=self.context_length,
                use_custom_token=False,
                need_create_context_pos=False
            ),
        }

    # pylint: disable=W0613
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        """apply conversion prompt"""
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        img_shape = result_recorder.get('images_shape')
        result_recorder.put("no_image_tag", [True])
        if img_shape:
            result_recorder.put("no_image_tag", [False])
        num_patches_list = [img_shape[0]] if img_shape is not None else []
        for (old_question, old_answer) in self.history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], ''.join(raw_inputs))
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = self.img_start_token \
                           + self.img_context_token * self.context_length * num_patches \
                           + self.img_end_token
            query = query.replace(self.context_pad_token, image_tokens, 1)
        return query

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

        batched_data["input_ids"] = batch_input_ids.astype(np.int32)
        batched_data["images"] = final_batch_images
        return batched_data

    # pylint: disable=W0613
    def post_process(self, output_ids, **kwargs):
        output = []
        for output_ids_item in output_ids:
            decoded = self.tokenizer.decode(output_ids_item, skip_special_tokens=True)
            output.append(decoded)
        return output
