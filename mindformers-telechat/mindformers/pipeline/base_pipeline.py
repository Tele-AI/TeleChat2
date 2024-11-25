# Copyright 2022-2024 Huawei Technologies Co., Ltd
# Copyright 2023 The HuggingFace Inc. team.
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
# https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py
# ============================================================================

"""BasePipeline"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import os
import numpy as np

from tqdm import tqdm
from mindspore import Tensor, Model
from mindspore.dataset import (
    GeneratorDataset, VisionBaseDataset,
    SourceDataset, MappableDataset
)
from mindspore.dataset.engine.datasets import BatchDataset, RepeatDataset

from mindformers.tools.logger import logger
from mindformers.mindformer_book import print_dict
from mindformers.models import BaseModel, BaseImageProcessor, PreTrainedTokenizerBase
from mindformers.models.modeling_utils import PreTrainedModel


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility
    """

    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    r"""
    Base Pipeline For All Task Pipelines

    Args:
        model (Union[PreTrainedModel]):
            The model used to perform task, the input should be a model instance inherited from PreTrainedModel.
        tokenizer (Optional[PreTrainedTokenizerBase]):
            The tokenizer of model, it could be None if the model do not need tokenizer.
        image_processor (Optional[BaseImageProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.
        framework (`str`, *optional*):
            The framework to use, only support `"ms"` for now. MindSpore framework must be installed.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
            Reversed for now.
    """
    _support_list = {}

    def __init__(self, model: Union[BaseModel, PreTrainedModel, Model],
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 feature_extractor=None,
                 image_processor: Optional[BaseImageProcessor] = None,
                 framework: Optional[str] = "ms",
                 task: str = "",
                 binary_output: bool = False,
                 **kwargs):
        super(Pipeline, self).__init__()
        self.model = model
        if isinstance(model, (BaseModel, PreTrainedModel)):
            self.network = model
        elif isinstance(model, Model):
            self.network = model.predict_network
        else:
            raise TypeError(f"model should be inherited from PreTrainedModel or Model, but got type {type(model)}.")
        self.framework = framework
        self.task = task
        self.binary_output = binary_output
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self._preprocess_params, self._forward_params, \
        self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0

        self._batch_size = kwargs.pop("batch_size", None)

    def __call__(self, inputs: Union[GeneratorDataset, list, str, Tensor, np.array],
                 batch_size: Optional[int] = None,
                 **kwargs):
        r"""Call Method

        Args:
            inputs (Union[GeneratorDataset, list, str, etc]):
            The inputs of pipeline, the type of inputs depends on task.
            batch_size (Optional[int]):
            The batch size for a GeneratorDataset input, for other types of inputs, the batch size would be set to 1 by
            default.

        Returns:
            outputs: The outputs of pipeline, the type of outputs depends on task.
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        is_dataset = isinstance(inputs, (
            GeneratorDataset, VisionBaseDataset,
            MappableDataset, SourceDataset))
        is_list = isinstance(inputs, list)

        self.call_count += 1
        if self.call_count == 20 and not is_dataset:
            logger.info("You seem to be using the pipeline sequentially for"
                        " numerous samples. In order to maximize efficiency"
                        " please set input a mindspore.dataset.GeneratorDataset.")

        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        if is_dataset:
            logger.info("dataset is processing.")
            if not isinstance(inputs, (BatchDataset, RepeatDataset)):
                inputs = inputs.batch(batch_size)

            outputs = []
            for items in tqdm(inputs.create_dict_iterator()):
                outputs.extend(self.run_single(items, preprocess_params,
                                               forward_params, postprocess_params))
        elif is_list:
            outputs = self.run_multi(inputs, batch_size, preprocess_params, forward_params, postprocess_params)
        else:
            outputs = self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

        return outputs

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int):
        if not isinstance(bs, int):
            raise ValueError('batch_size must be an integer!')
        if bs < 0:
            raise ValueError('batch_size must be positive!')
        self._batch_size = bs

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    def save_pretrained(self, save_directory: str, save_name: str = 'mindspore_model'):
        r"""Save the pipeline's model and tokenizer

        Args:
            save_directory('str'):
                A path to the directory where to saved. It will be created if it doesn't exist
            save_name(str): the name of saved files, including model weight and configuration file.
                Default mindspore_model.
        """
        if os.path.isfile(save_directory):
            logger.error(f"provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory, save_name)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory, save_name=save_name)

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory, save_name)

        # save image_processor dep
        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory, save_name)

    def transform(self, *args, **kwargs):
        """Compatibility method
        Scikit / Keras interface to pipelines. This method will forward to __call__()
        """
        return self(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Compatibility method
        Scikit / Keras interface to pipelines. This method will forward to __call__()
        """
        return self(*args, **kwargs)

    def run_single(self, inputs: Union[dict, str, np.array, Tensor],
                   preprocess_params: dict,
                   forward_params: dict,
                   postprocess_params: dict):
        r"""Run Single method
        This function is used to run a single forward process for task.

        Args:
            inputs (Union[dict, str, etc]):
                The inputs of pipeline, the type of inputs depends on task.
            preprocess_params (dict):
                The parameter dict for preprocess.
            forward_params (dict):
                The parameter dict for model forward process.
            postprocess_params (dict):
                The parameter dict for postprocess.
        """
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def run_multi(self, inputs: Union[list, tuple],
                  batch_size: int,
                  preprocess_params: dict,
                  forward_params: dict,
                  postprocess_params: dict):
        r"""Run Multiple Method
        This function is used to run a list input for task.

        Args:
            inputs (Union[list, tuple, iterator]):
                The iterable input for pipeline.
            batch_size (int):
                Batch size of pipeline input.
            preprocess_params (dict):
                The parameter dict for preprocess.
            forward_params (dict):
                The parameter dict for model forward process.
            postprocess_params (dict):
                The parameter dict for postprocess.
        """
        if len(inputs) % batch_size != 0:
            raise ValueError(f"When running multi input pipeline, the length of inputs {len(inputs)}"
                             f" should be multiple of batch size {batch_size}. Please check yout inputs.")
        outputs = []
        if batch_size > 1:
            batch_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        else:
            batch_inputs = inputs
        for item in batch_inputs:
            outputs.extend(self.run_single(item, preprocess_params,
                                           forward_params, postprocess_params))
        return outputs

    @abstractmethod
    def preprocess(self, inputs: Union[dict, str, np.array, Tensor],
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[dict, str, etc]):
                The inputs of pipeline, the type of inputs depends on task.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("preprocess not implemented.")

    def forward(self, model_inputs: Union[dict, str, np.array, Tensor],
                **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (Union[dict, str, etc]):
                The output of preprocess, the type of model_inputs depends on task.
            forward_params (dict):
                The parameter dict for model forward.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        return self._forward(model_inputs, **forward_params)

    @abstractmethod
    def _forward(self, model_inputs, **forward_params):
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: Union[dict, str, np.array, Tensor],
                    **postprocess_params):
        r"""The Postprocess of Task

        Args:
            model_outputs (Union[dict, str, etc]):
                The output of model forward, the type of model_outputs depends on task.
            postprocess_params (dict):
                The parameter dict for post process.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("postprocess not implemented.")

    @classmethod
    def show_support_list(cls):
        """show_support_list"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)
