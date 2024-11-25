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
"""Transform for QwenVL"""

import numpy as np
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from qwenvl_tokenizer import IMG_TOKEN_SPAN

# ignore token id 根据输入
IGNORE_TOKEN_ID = -100


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class QwenVLTransform:
    """
    Caption Transform, preprocess captions and tokenize it,
    align with torch impl.
    """

    def __init__(self, tokenizer,
                 prompt=None,
                 max_img_size=IMG_TOKEN_SPAN,
                 padding="max_length",
                 max_length=512,
                 max_annotation=None,
                 random_seed=2022, truncation=True, add_special_tokens=True):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_img_size = max_img_size
        self.max_length = max_length
        self.padding = padding
        self.random_seed = random_seed
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.max_annotation = max_annotation
        if prompt is None:
            prompt = 'Describe the image in English'
        self.img_padding = self.tokenizer.image_pad_tag * self.max_img_size
        placeholder = '{}'
        self.template = {
            'caption': f'<img></img>{prompt}: {placeholder}',
            'vqa': f'<img></img>{placeholder} Answer: {placeholder}',
        }

    def __call__(self, caption, template=None):
        if template is None:
            template = self.template

        out = self.pre_caption(caption, template)
        if len(out) == 2:
            cap_out, img_start_pos = out
            cap_out = np.stack(cap_out).astype(np.int32)
            img_start_pos = np.stack(img_start_pos).astype(np.int32)
            return cap_out, img_start_pos

        cap_out, img_start_pos, label = out
        cap_out = np.stack(cap_out).astype(np.int32)
        img_start_pos = np.stack(img_start_pos).astype(np.int32)
        label = np.stack(label).astype(np.int32)
        return cap_out, img_start_pos, label

    def pre_caption_for_caption_task(self, caption, template):
        """process caption for caption_task"""
        task = 'caption'
        if isinstance(caption, dict):
            caption = caption[task]
        if isinstance(caption, list) and (len(caption) > 1):
            caption_list = [self.pre_caption(cap, template) for cap in caption]
            caption_list = list(zip(*caption_list))
            return caption_list
        caption = template[task].format(caption)
        return self.tokenizer(caption, max_length=self.max_length, padding=self.padding)["input_ids"]

    def pre_caption_for_vqa_task(self, caption, template):
        """process caption for caption_task"""
        task = 'vqa'
        if isinstance(caption['answers'], list) and (len(caption) > 1):
            caption_list = []
            for ans in caption['answers']:
                qa = {'question': caption['question'], 'answers': ans}
                caption_list.append(self.pre_caption(qa, template))
            caption_list = list(zip(*caption_list))
            return caption_list
        question = caption['question']
        caption = caption['answers']
        caption = template[task].format(question, caption)
        return self.tokenizer(caption, max_length=self.max_length, padding=self.padding)["input_ids"]

    def generate_input_and_label_for_sft_data(self, caption, raw_data, raw_data_role):
        """generate input ids and labels for sft data"""
        input_ids = []
        labels = []
        user_role_name = 'user' if caption.get('user_role_name') is None else caption.get('user_role_name')
        for i, cap in enumerate(raw_data):
            img_pad_token_id = self.tokenizer.image_pad_tag * self.max_img_size
            cap = cap.replace('{}', img_pad_token_id)
            tokenized_cap = self.tokenizer(cap)['input_ids']
            input_ids.extend(tokenized_cap)

            if raw_data_role[i] == user_role_name or raw_data_role[i] == 'system':
                ignore_token = [IGNORE_TOKEN_ID] * (len(tokenized_cap) - 3)
                labels.extend(self._add_start_end_label(ignore_token))
            elif raw_data_role[i] == 'assistant' if caption.get(
                    'assistant_role_name') is None else caption.get('user_role_name'):
                data_role_input_ids = self.tokenizer('<|im_start|>' + raw_data_role[i])['input_ids']
                has_ignored_label = [IGNORE_TOKEN_ID] * len(data_role_input_ids) \
                                    + tokenized_cap[len(data_role_input_ids) + 1:-2]
                labels.extend(self._add_start_end_label(has_ignored_label))

            else:
                raise ValueError(f"raw_data_role {raw_data_role[i]} is invalid")
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels += [IGNORE_TOKEN_ID] * (self.max_length - len(labels))
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        return input_ids, labels

    def generate_img_pos(self, input_ids, img_idx):
        """padding and generate coord for image"""
        new_img_idx = []
        img_start_temp = []
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.img_start_id:
                img_start_temp.append(i + 1)

        if not img_start_temp:
            new_img_idx = [self.max_length - self.max_img_size - 1] * len(img_idx)
        else:
            for i, idx in enumerate(img_idx):
                if idx != -1:
                    new_img_idx.append(img_start_temp[i])
                else:
                    new_img_idx.append(self.max_length - self.max_img_size - 1)
        return self._generate_coord(new_img_idx)

    def pre_caption(self, caption, template):
        """
        Caption preprocessing removes any punctuation marks except commas,
        tailing spaces and transform sentence into lower case.
        """
        task = caption.get('task')
        if task is None:
            raise ValueError("task is required in the output of dataloader")

        if task == 'sft':
            raw_data = caption.get('raw_data')
            raw_data_role = caption.get('raw_data_role')
            img_idx = caption.get('img_idx')

            if len(img_idx) * self.max_img_size > self.max_length - 1:
                raise ValueError(
                    f"Sum of query token length of max_img_size images has exceeds the max_length of text. "
                    f"max_img_size={self.max_img_size}, max_length={self.max_length}")

            if raw_data is None or raw_data_role is None or img_idx is None:
                raise ValueError("raw_data, raw_data_role and img_idx are required")

            input_ids, labels = self.generate_input_and_label_for_sft_data(caption, raw_data, raw_data_role)
            coord = self.generate_img_pos(input_ids, img_idx)
            return input_ids, coord, labels

        if task == 'vqa':
            input_ids = self.pre_caption_for_vqa_task(caption, template)
        else:
            input_ids = self.pre_caption_for_caption_task(caption, template)

        input_ids = np.array(input_ids, dtype=np.int32)
        img_start_pos = np.where(input_ids == self.tokenizer.img_start_id)[0] + 1
        coord = self._generate_coord(img_start_pos)
        return input_ids, coord

    def _add_start_end_label(self, input_list):
        im_start = self.tokenizer.im_start_id
        im_end = self.tokenizer.im_end_id
        nl_token_id = self.tokenizer('\n')["input_ids"]
        input_list = [im_start] + input_list + [im_end] + nl_token_id
        return input_list

    def _generate_coord(self, img_start_pos):
        num_img = len(img_start_pos)
        coord = np.zeros((num_img, self.max_img_size, 2), np.int32)
        for idx, pos in enumerate(img_start_pos):
            for img_pos in range(self.max_img_size):
                coord[idx, img_pos] = [0, pos + img_pos]
        return coord
