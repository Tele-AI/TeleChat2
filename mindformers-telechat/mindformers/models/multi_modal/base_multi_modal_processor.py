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
BaseImageToTextProcessor
"""
from typing import Optional, Union, List, Dict

import PIL
import PIL.Image
import numpy as np
import mindspore as ms
from mindspore.dataset import vision

from mindformers.dataset import build_transforms
from mindformers.dataset.transforms.vision_transforms import BatchPILize, BatchResize, BatchToTensor, BatchNormalize
from mindformers.models.base_processor import BaseProcessor
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.multi_modal.modal_content import ModalContentTransformTemplate
from mindformers.tools import MindFormerModuleType, MindFormerRegister
from .utils import DataRecord


class BatchResizeV2(BatchResize):
    """
    Resize a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
         interpolation (str): interpolate method, default is 'cubic'.
    """

    def __init__(self, image_resolution, interpolation='cubic'):
        super().__init__(image_resolution, interpolation)
        self.resize = vision.Resize(image_resolution, self.interpolation)


class BaseImageToTextImageProcessor(BaseImageProcessor):
    """
    BaseImageToTextImageProcessor.

    Args:
        image_size (int): The target size.
    """

    def __init__(self,
                 image_size: Optional[int] = 224,
                 interpolation: Optional[str] = 'bicubic',
                 mean=None,
                 std=None,
                 is_hwc=False,
                 **kwargs):
        self.pilize = BatchPILize()
        super().__init__(**kwargs)
        if isinstance(image_size, int):
            image_size = (image_size,) * 2
        self.resize = BatchResizeV2(image_size, interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = BatchNormalize(mean, std, is_hwc)

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = self.pilize(images)
        images = self.resize(images)
        images = self.to_tensor(images)
        images = self.normalize(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    @staticmethod
    def _bhwc_check(image_batch: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray) and image_batch.shape[-1] == 3:
            return True
        if isinstance(image_batch, ms.Tensor) and image_batch.asnumpy().shape[-1] == 3:
            return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BaseXModalToTextTransform:
    """
        Base multi-modal to text transform, it can perform transforms defined in the template in a order.
    """
    def __init__(
            self,
            tokenizer,
            model_transform_template: ModalContentTransformTemplate,
            max_length=512,
            padding='max_length',
            mode: str = "train",
    ):
        self.tokenizer = tokenizer

        if isinstance(model_transform_template, dict):
            model_transform_template = build_transforms(
                model_transform_template, default_args={"tokenizer": tokenizer, "max_length": max_length})

        self.model_transform_template = model_transform_template

        self.max_length = max_length
        self.padding = padding
        self.mode = mode

        if not self.model_transform_template.has_init_modal_builder_tokens:
            self.model_transform_template.check_modal_builder_tokens(self.tokenizer)

        self.modal_builders = model_transform_template.modal_builders

        self.result_recorder = DataRecord()

    def update_result_before_output(self):
        update_items = self.model_transform_template.get_need_update_output_items(self.result_recorder)
        self.result_recorder.put_from_dict(update_items)

    def perform_predict_transform(self, query_ele_list: List[Dict], batch_index=0, **kwargs):
        """perform transform for prediction"""
        self.result_recorder.clear()
        add_special_tokens = kwargs.get("add_special_tokens", False)
        text_list = self.model_transform_template.process_predict_query(query_ele_list, self.result_recorder)
        text = self.model_transform_template.build_conversation_input_text(text_list, self.result_recorder)

        text_id = self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        text_id = self.model_transform_template.build_modal_context(text_id, self.result_recorder, **kwargs)

        context_pos_dict = self.model_transform_template.generate_modal_context_positions(text_id, batch_index,
                                                                                          self.result_recorder,
                                                                                          **kwargs)
        self.result_recorder.put("raw_query", text)
        self.result_recorder.put("input_ids", text_id)
        self.result_recorder.put_from_dict(context_pos_dict)

        self.update_result_before_output()

        output_columns = ["raw_query"] + self.model_transform_template.output_columns
        return self.result_recorder.output(output_columns, format_="dict")

    def batch_for_predict(self, batch_data, token_padding_length=None, **kwargs):
        return self.model_transform_template.batch(batch_data, token_padding_length, **kwargs)

    def post_process_for_predict(self, output_ids, **kwargs):
        return self.model_transform_template.post_process_for_predict(output_ids, **kwargs)

    def perform_train_transform(self, conversations, **kwargs):
        """perform transform for training"""
        self.result_recorder.clear()
        text_list = self.model_transform_template.process_train_item(conversations, self.result_recorder)
        text_list = self.model_transform_template.build_conversation_input_text(text_list, self.result_recorder)

        text_id_list = []
        for text in text_list:
            text_id = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            text_id = self.model_transform_template.build_modal_context(text_id, self.result_recorder, **kwargs)
            text_id_list.append(text_id)

        labels = self.model_transform_template.build_labels(text_id_list, self.result_recorder, **kwargs)
        concat_text_id = np.concatenate(text_id_list)
        context_pos_dict = self.model_transform_template.generate_modal_context_positions(concat_text_id, **kwargs)

        text_id, labels = self.padding_input_ids_and_labels_for_train(concat_text_id, labels)
        self.result_recorder.put("input_ids", text_id.astype(np.int32))
        self.result_recorder.put("labels", labels.astype(np.int32))
        self.result_recorder.put_from_dict(context_pos_dict)

        self.update_result_before_output()

        output_columns = self.model_transform_template.output_columns
        return self.result_recorder.output(output_columns, format_="tuple")

    def padding_input_ids_and_labels_for_train(self, text_ids, labels):
        """padding input_ids and labels to max_length"""
        if isinstance(text_ids, list):
            text_ids = np.array(text_ids)

        if isinstance(labels, list):
            labels = np.array(labels)

        cur_length = len(text_ids)
        target_length = self.max_length + 1

        if cur_length < target_length:
            padding_length = target_length - cur_length
            text_ids = np.pad(text_ids, (0, padding_length), "constant", constant_values=self.tokenizer.pad_token_id)
            labels = np.pad(labels, (0, padding_length), "constant", constant_values=-100)
        else:
            text_ids = text_ids[:target_length]
            labels = labels[:target_length]
        return text_ids, labels

    def post_process(self, output_ids, **kwargs):
        return self.model_transform_template.post_process(output_ids, **kwargs)

    def __call__(self, conversations, **kwargs):
        if conversations.shape == ():
            return self.perform_train_transform(self.process_conversation(conversations))
        return self.perform_train_transform(conversations)

    def process_conversation(self, conversations):
        """process data"""
        import json
        dict_data = json.loads(np.array_str(conversations))

        conversation_data = []
        for message in dict_data:
            from_ = message["from"]
            value = message["value"]
            conversation_data.append([from_, value])
        return conversation_data


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class BaseXModalToTextProcessor(BaseProcessor):
    r"""BaseXModalToTextProcessor,
    consists of a feature extractor (BaseFeatureEXtractor) for multi-modal input,
    and a tokenizer for text input.

    Args:
        image_processor (BaseImageProcessor): Used for process image data.
        tokenizer: Used for process text data.
        max_length (Optional[int]): The length of text tokens.
        padding (Optional[str]): The padding strategy of tokenizer, [None, "max_length"].
        return_tensors (Optional[str]): The type of returned tensors for tokenizer, [None, "ms"].
    """

    def __init__(self, model_transform_template: ModalContentTransformTemplate,
                 tokenizer,
                 max_length=None,
                 padding='max_length',
                 **kwargs):
        super().__init__(
            model_transform_template=model_transform_template,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding, **kwargs)

        self.modal_transform = BaseXModalToTextTransform(tokenizer,
                                                         model_transform_template,
                                                         mode="predict")
        self.kwargs = kwargs
        self.output_columns = model_transform_template.output_columns

    def post_process(self, output_ids, **kwargs):
        """post process the predict results"""
        return self.modal_transform.post_process(output_ids, **kwargs)

    # pylint: disable=W0222
    def __call__(self, query_list: Union[List[Dict], List[List[Dict]]], history_list=None):
        """call function"""
        if isinstance(query_list, list) and query_list and isinstance(query_list[0], dict):
            query_list = [query_list]

        batch_data = []
        history_list = None

        max_length = self.max_length or 0
        for index, query in enumerate(query_list):
            if history_list is not None:
                history = history_list[index]
            else:
                history = None
            data = self.modal_transform.perform_predict_transform(query, batch_index=index, history=history,
                                                                  **self.kwargs)
            batch_data.append(data)
            max_length = max(max_length, len(data.get("input_ids")))

        return self.modal_transform.batch_for_predict(batch_data, max_length)
