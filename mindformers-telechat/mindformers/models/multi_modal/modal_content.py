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
ModalContent Tools
"""
import abc
import os
import re
from typing import Dict, Any, List

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

from mindformers.tools.logger import logger
from .utils import DataRecord
from ...tools.image_tools import load_image


class ModalContentBuilder:
    """
    Base class of builder for modal content. Modal content builder processes the input into what the model expects in
    different phase.
    """
    def __init__(
            self, type_, context_pad_token,
            context_length,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            need_create_context_pos=True,
            need_padding_context=True,
            modal_content_max_size=1,
            mode="predict",
            max_length=2048
    ):
        self.type = type_
        self.context_pad_token = context_pad_token
        self.context_length = context_length
        self.use_custom_token = use_custom_token

        if not self.use_custom_token and (not start_token or not end_token) and type_ != "text":
            raise ValueError("start_token and end_token must be set when use_custom_token is False")

        if self.use_custom_token and not start_token:
            start_token = f"<|MF_CUSTOM_{type_.upper()}_START|>"

        if self.use_custom_token and not end_token:
            end_token = f"<|MF_CUSTOM_{type_.upper()}_END|>"

        self.start_token = start_token
        self.end_token = end_token

        self.tokenizer = tokenizer
        self.need_create_context_pos = need_create_context_pos
        self.need_padding_context = need_padding_context
        self.modal_content_max_size = modal_content_max_size
        self.mode = mode
        self.max_length = max_length

        if mode not in ["train", "predict"]:
            raise ValueError(f"only mode `train` and `predict` are supported, {mode} is got.")

        self.has_add_custom_token = False
        self.start_token_id = None
        self.end_token_id = None
        self.context_pad_token_id = None

    @abc.abstractmethod
    def regular_input_for_predict(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        start_position = np.where(input_ids == self.start_token_id)[0]
        offset = 0
        for start_position_item in start_position:
            start_position_item = start_position_item + 1 + offset

            input_ids = np.insert(input_ids, start_position_item, [self.context_pad_token_id] * self.context_length)
            offset += self.context_length

        if self.use_custom_token:
            input_ids = np.delete(input_ids, np.where(input_ids == self.start_token_id))
            input_ids = np.delete(input_ids, np.where(input_ids == self.end_token_id))

        return input_ids

    def generate_context_positions(self, input_ids, batch_index=0, result_recorder: DataRecord = None, **kwargs):
        """generate context positions in the text, it will pad fake positions in the training task"""
        pos = np.where(np.array(input_ids) == self.context_pad_token_id)[0]

        if self.mode == "train":
            if pos.shape[0] == 0:
                pos = np.tile(np.array(list(range(self.max_length - self.context_length, self.max_length))),
                              self.modal_content_max_size)
            elif len(pos) <= self.modal_content_max_size * self.context_length:
                padding_size = (self.modal_content_max_size * self.context_length - len(pos)) // self.context_length
                padding_array = np.array(list(range(self.max_length - self.context_length, self.max_length)))
                pos = np.concatenate([pos, np.tile(padding_array, padding_size)])
            else:
                pos = pos[:self.modal_content_max_size * self.context_length]
                logger.warning("context pos items is greater than modal_content_max_size, it will be truncated.")

        pos = np.expand_dims(pos, axis=0)
        pos = np.insert(pos, 0, batch_index, axis=0)
        pos = np.transpose(pos).reshape((-1, self.context_length, 2))
        return pos

    @staticmethod
    def find_tag_and_extract(text, start_tag, end_tag, check_close=True):
        """find specific tag in the text and extract content in the tags"""
        start_tag_index = []
        end_tag_index = []
        for match in re.finditer(re.escape(start_tag), text):
            start_tag_index.append((match.start(), match.end()))

        for match in re.finditer(re.escape(end_tag), text):
            end_tag_index.append((match.start(), match.end()))

        if check_close and len(start_tag_index) != len(end_tag_index):
            raise ValueError(f"the text has unclosed {start_tag}{end_tag} in {text}")

        replaced_text = []
        content = []
        last_end = 0
        for start_tag_index_item, end_tag_index_item in zip(start_tag_index, end_tag_index):
            start_tag_start_idx, start_tag_end_idx = start_tag_index_item
            end_tag_start_idx, end_tag_end_idx = end_tag_index_item

            if start_tag_end_idx > end_tag_start_idx:
                raise ValueError(f"the `text` {text} has error start and end tag order")

            replaced_text.append(text[last_end:start_tag_start_idx])
            content.append(text[start_tag_end_idx:end_tag_start_idx])
            last_end = end_tag_end_idx

        replaced_text.append(text[last_end:])

        tag_padding = f"{start_tag}{end_tag}"
        padded_text = tag_padding.join(replaced_text)
        return padded_text, content


class BaseTextContentBuilder(ModalContentBuilder):
    """
    Base text modal builder. It returns the original input.
    """

    def __init__(self):
        super().__init__(type_="text",
                         context_pad_token="",
                         context_length=-1,
                         use_custom_token=False,
                         start_token="",
                         end_token="",
                         need_create_context_pos=False)

    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        if "text" in inputs:
            return inputs["text"]
        return ""

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        return inputs


class BaseImageContentBuilder(ModalContentBuilder):
    """
    Base image modal builder, it returns the padded input in training task after calling
    method `padding_images_to_max_content_size`
    """
    def __init__(
            self,
            context_pad_token,
            context_length,
            image_size=448,
            image_location="",
            use_custom_token=False,
            start_token="<img>",
            end_token="</img>",
            tokenizer=None,
            need_padding_context=True,
            modal_content_max_size=1,
            mode="train",
            max_length=2048
    ):
        super().__init__(
            type_="image",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
            need_padding_context=need_padding_context,
            modal_content_max_size=modal_content_max_size,
            mode=mode,
            max_length=max_length
        )
        self.image_size = image_size
        self.image_location = image_location
        self.image_mapping = None

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        text, image_path_list = self.find_tag_and_extract(inputs, self.start_token, self.end_token)
        images = self.load_images(image_path_list)
        result_recorder.put("images", images, append=True)
        return text

    def regular_input_for_predict(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError

    def load_images(self, image_path_list):
        """load images"""
        image_list = []
        for image_path in image_path_list:
            image = np.array(load_image(os.path.join(self.image_location, image_path)))
            if self.image_mapping is not None and callable(self.image_mapping):
                # pylint: disable=E1102
                image = self.image_mapping(image)
            image_list.append(image)
        return np.stack(image_list)

    def padding_images_to_max_content_size(self, images):
        """padding images to max content size"""
        if images is None:
            padding_shape = (self.modal_content_max_size, self.image_size, self.image_size, 3)
            padded_images = np.ones(padding_shape, dtype=np.uint8)
        else:
            images = np.vstack(images)
            cur_image_len = images.shape[0]
            if cur_image_len == self.modal_content_max_size:
                padded_images = images
            elif cur_image_len < self.modal_content_max_size:
                padding_images = np.tile(images[-1], (self.modal_content_max_size - cur_image_len, 1, 1, 1))
                padded_images = np.vstack([images, padding_images])
            else:
                logger.warning(f"The number of images is greater than the `modal_content_max_size`, {cur_image_len} > "
                               f"{self.modal_content_max_size}. it will reserve {self.modal_content_max_size} images.")
                padded_images = images[:self.modal_content_max_size]
        return padded_images


class ModalContentTransformTemplate:
    """
    Base class of modal content transform template. It should be implemented by the specific model. The child class can
    override the methods `build_conversion_input_text`, `update_result_before_output`, `batch`, `post_process` to
    achieve the model's expectations.

    Args:
        output_columns (List[str], optional): Specify which columns will be output. Default: ``None`` .
        tokenizer (Tokenizer, optional): Build a good model tokenizer. Default: ``None`` .
        mode (str): running mode, predict or train. Default: ``predict`` .
        vstack_columns (List[str], optional): Specify which columns will be vstack when batching data.
            Default: ``None`` .
        modal_content_padding_size (int): Used in training mode for inherited Template subclasses,
            it usually represents the maximum number of
            supported modal contents (such as images) within a training sample.
            When the number of modal contents in a training sample is less than this value,
            the modal contents will be expanded to that value.
        max_length (int): Used in training mode, for inherited Template subclasses,
            it usually represents the maximum length that a training sample
            can fill in after the content mask is completed after segmentation.
        kwargs (dict, optional): A variable number of keyword parameters reserved
            for the keyword parameters to be expanded.

    Examples:
        >>> from mindformers.models.multi_modal import ModalContentTransformTemplate
        >>> ModalContentTransformTemplate().supported_modal
        []
        >>> # Note:
        >>> #     The property of 'supported_modal' should be inherited by subclasses,
        >>> #     and subclasses implement the corresponding modal builders.
        >>> #     The current base class does not support any modal builders, so it returns '[]'.
    """
    _DEFAULT_OUTPUT_COLUMNS = ["input_ids"]

    # pylint: disable=W0613
    def __init__(self, output_columns: List[str] = None, tokenizer=None, mode="predict",
                 vstack_columns: List[str] = None, modal_content_padding_size=1, max_length=2048,
                 **kwargs):
        if output_columns is None:
            self.output_columns = self._DEFAULT_OUTPUT_COLUMNS
        else:
            self.output_columns = output_columns

        self.tokenizer = tokenizer
        self.mode = mode
        self.modal_content_padding_size = modal_content_padding_size

        if vstack_columns is not None:
            new_vstack_columns = [column for column in vstack_columns if column in self.output_columns]

            if len(new_vstack_columns) != len(vstack_columns):
                logger.warning("there are columns in vstack_columns not in output_columns, it will be ignored.")
            vstack_columns = new_vstack_columns
        else:
            vstack_columns = []
        self.vstack_columns = vstack_columns

        self.history = []
        self.modal_builders: Dict[str, ModalContentBuilder] = {}
        self._supported_modal = []
        self._supported_modal_tag = []
        self.max_length = max_length

        self.has_init_modal_builder_tokens = False
        if self.tokenizer is not None:
            self.check_modal_builder_tokens(self.tokenizer)

        self.tensor_stack = ops.Stack(axis=0)
        self.tensor_vstack = ops.vstack

    def process_predict_query(self, query_ele_list: List[Dict], result_recorder: DataRecord):
        """
        In predict mode, find the corresponding modal builder by traversing and process it.

        Args:
            query_ele_list (List[dict]): A list of elements for predicting a request.
                For example: [{"image":"/path/to/image"}, {"text":"describe image in English"}].
            result_recorder (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            The text results processed by each modal builder.
        """
        text_list = []
        for query_ele in query_ele_list:
            modal_type = ""
            for supported_modal_type in self.supported_modal:
                if supported_modal_type in query_ele:
                    modal_type = supported_modal_type
                    break

            if not modal_type:
                raise ValueError(f"The modal_type of {query_ele} is not supported by current model,"
                                 f"please check the predict input.")

            text_list.append(
                self.modal_builders.get(modal_type).regular_input_for_predict(
                    inputs=query_ele, result_recorder=result_recorder))
        return text_list

    def process_train_item(self, conversation_list: List[List], result_recorder: DataRecord):
        """
        In train mode, find the corresponding modal builder by traversing and process it.

        Args:
            conversation_list (List[List]): A list of elements for dialogue data.
                For example: [["user", "<img>/path/to/image<img>describe image in English:"],
                ["assistant", "the image describe ...."]]
            result_recorder (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            The text results processed by each modal builder.
        """
        text_list = []
        for key_from, conversation in conversation_list:
            modal_type = ""
            for supported_modal_type in self.supported_modal:
                modal_start_tag = self.modal_builders.get(supported_modal_type).start_token
                if modal_start_tag in conversation:
                    modal_type = supported_modal_type
                    break

            if not modal_type:
                logger.warning("The %s is not recognized by any modal_builders, it will be regard as pure text",
                               conversation)
                modal_type = "text"

            text_list.append(
                [key_from, self.modal_builders.get(modal_type).regular_input_for_train(
                    inputs=conversation, result_recorder=result_recorder)])
        return text_list

    @property
    def supported_modal(self):
        """
        Used to return the templates supported of modal builder type by an instance.

        Returns:
            List type, containing the types of modal builder supported by an instance.
        """
        if not self._supported_modal:
            self._supported_modal = [modal_builder.type for modal_builder in self.modal_builders.values()]
        return self._supported_modal

    @abc.abstractmethod
    def build_conversation_input_text(self, raw_inputs, result_recorder: DataRecord):
        """
        Used in predict mode, assemble a conversation based on incoming inputs.
        Usually inherited and used by quilt class.

        Args:
            raw_inputs (str): input data.
            result_recorder (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            Str type. Assembled dialogue.
        """
        if self.mode == "predict":
            return "".join(raw_inputs)
        raise ValueError(f"building {self.mode} mode conversion inputs is not supported.")

    def build_modal_context(self, input_ids, result_recorder: DataRecord, **kwargs):
        """
        According to the requirements of the modal builder,
        process the input_ids and finally return the processed input_ids.

        Args:
            input_ids (list): input data.
            result_recorder (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.

        Returns:
            The processed input_ids.
        """
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        for modal_builder in self.modal_builders.values():
            if not modal_builder.need_padding_context:
                continue
            input_ids = modal_builder.padding_context(input_ids, result_recorder, **kwargs)
        return input_ids

    def build_labels(self, text_id_list, result_recorder, **kwargs):
        """
        Used in training mode, for subclasses to inherit, to construct the labels needed for training from text data.

        Args:
            text_id_list (list): A list containing text data identifiers or indices.
            result_recorder (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.
        """
        # pylint: disable=W0107
        pass

    def generate_modal_context_positions(self, input_ids, batch_index: int = 0,
                                         result_recorder: DataRecord = None, **kwargs):
        """generate modal context positions in the text by traversing all modal builders"""
        context_positions = {}
        for modal_builder in self.modal_builders.values():
            if not modal_builder.need_create_context_pos:
                continue

            context_pos = modal_builder.generate_context_positions(input_ids, batch_index=batch_index,
                                                                   result_recorder=result_recorder, **kwargs)
            context_pos = context_pos.astype(np.int32)
            if self.mode == "predict":
                context_pos = Tensor(context_pos, dtype=ms.int32)

            context_positions[f"{modal_builder.type}_context_pos"] = context_pos
        return context_positions

    def check_modal_builder_tokens(self, tokenizer):
        """check modal builder tokens status, it will be assigned when not inited."""
        for modal_builder in self.modal_builders.values():
            if modal_builder.use_custom_token:
                tokenizer.add_tokens([modal_builder.start_token, modal_builder.end_token], special_tokens=True)

            if modal_builder.type == "text":
                continue

            if modal_builder.start_token_id is None:
                modal_builder.start_token_id = \
                    tokenizer([modal_builder.start_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.end_token_id is None:
                modal_builder.end_token_id = \
                    tokenizer([modal_builder.end_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.context_pad_token_id is None:
                modal_builder.context_pad_token_id = \
                    tokenizer([modal_builder.context_pad_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.tokenizer is None:
                modal_builder.tokenizer = tokenizer
            self.has_init_modal_builder_tokens = True

    def get_need_update_output_items(self, result: DataRecord) -> Dict[str, Any]:
        """
        Retrieve the output items that need to be updated.

        Args:
            result (DataRecord): The result data recorder is used to save data that
                needs to be recorded during the inference process.
                Values are stored by calling the put method of the DataRecord.

        Returns:
            A Dict. Defaults to an empty dict.
        """
        update_items = {}
        return update_items

    def batch_input_ids(self, input_ids_list, max_length):
        """batch the input_ids_list with max_length, if the length of input_ids is greater than max_length, it will be truncated to the max_length"""
        padded_input_ids = []
        for input_ids in input_ids_list:
            if len(input_ids) < max_length:
                input_ids = np.pad(input_ids, (0, max_length - len(input_ids)), "constant",
                                   constant_values=self.tokenizer.pad_token_id)
            else:
                input_ids = input_ids[:max_length]

            padded_input_ids.append(input_ids)
        return np.stack(padded_input_ids, axis=0)

    def stack_data(self, data, need_vstack: bool = False):
        """stack data"""
        if isinstance(data[0], Tensor):
            if need_vstack:
                stacked_data = self.tensor_vstack(data)
            else:
                stacked_data = self.tensor_stack(data)
        else:
            if need_vstack:
                stacked_data = np.vstack(data)
            else:
                stacked_data = np.stack(data, axis=0)
        return stacked_data

    def try_to_batch(self, data_list, column_name):
        """try to batch data list"""
        batched_data = data_list
        need_vstack = column_name in self.vstack_columns
        try:
            batched_data = self.stack_data(data_list, need_vstack)
            batch_result = True
        # pylint: disable=W0703
        except Exception as e:
            logger.warning("batching %s failed, Error: %s.", column_name, e)
            batch_result = False

        return batch_result, batched_data

    def batch(self, data_list, token_padding_length, **kwargs):
        """
        Batch the column data in the output_names.

        Args:
            data_list (list): A list containing multiple data items.
            token_padding_length (int): Used to pad the length of "tokens" to ensure that
                all text data has the same length.
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.

        Returns:
            A dict. Used to store the batched data.
        """
        batched_data = {}
        for column_name in self.output_columns:
            column_data_list = [data[column_name] for data in data_list if
                                data[column_name] is not None and len(data[column_name]) > 0]

            if column_name == "input_ids":
                batched_data[column_name] = self.batch_input_ids(column_data_list, token_padding_length)
                continue

            if column_data_list:
                batch_success, batched_column_data = self.try_to_batch(column_data_list, column_name)
                batched_data[column_name] = batched_column_data
                if not batch_success:
                    logger.warning("batching %s failed, try to create a separate process for this data.", column_name)
            else:
                batched_data[column_name] = None
                logger.warning(f"the data {column_name} is empty when batching data, please check the query data.")
        return batched_data

    def post_process(self, output_ids, **kwargs):
        """
        Decode the model's output_ids into text strings.

        Args:
            output_ids (list): A list containing the model's output_ids.
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.

        Returns:
            A list containing all decoded text strings.
        """
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        output = []
        for output_ids_item in output_ids:
            output.append(self.tokenizer.decode(output_ids_item, skip_special_tokens))
        return output
