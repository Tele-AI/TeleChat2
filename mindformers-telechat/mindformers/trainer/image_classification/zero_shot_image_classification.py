# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Image Classification Trainer."""
from typing import Optional, List, Union

import numpy as np
from PIL.Image import Image

from mindspore.train import Callback
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase, BaseImageProcessor
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ...dataset.dataloader import build_dataset_loader
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer

__all__ = ['ZeroShotImageClassificationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class ZeroShotImageClassificationTrainer(BaseTrainer):
    """
    Trainer of zero-shot image classification task. It provides evaluation and prediction interfaces for
    zero-shot image classification task, allowing users to quickly start the process according to the model name,
    and also provides a large number of customizable items to meet user needs.

    Args:
        model_name (str): The model name of zero-shot image classification task trainer. Default: None

    Raises:
        NotImplementedError: If train method, evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(ZeroShotImageClassificationTrainer, self).__init__("zero_shot_image_classification", model_name)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "The ZeroShotImageClassification task does not support train.")

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                 network: Optional[Union[Cell, PreTrainedModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """
        The evaluation API of zero-shot image classification task. It allows to quickly start evaluation based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        dataset, callbacks, compute_metrics, and callbacks.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The evaluate dataset.
                It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It supports CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It supports dict or set in MindSpore's Metric class.
                Default: None.

        Returns:
            None
        """
        metric_name = "Top1 Accuracy"
        kwargs.setdefault("metric_name", metric_name)
        self.evaluate_process(
            config=config,
            network=network,
            dataset=dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[GeneratorDataset,
                                           Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """
        The prediction API of zero-shot image classification task. It allows to quickly start prediction based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        input data, tokenizer and the processor of image.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            input_data (Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]]):
                The dataset. It supports real dataset path or
                BaseDateset class or MindSpore Dataset class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]): Used for text process.
            image_processor (Optional[BaseImageProcessor]): Used for image process.

        Returns:
            A list of prediction results.
        """
        config = self.set_config(config)

        logger.info(".........Build Input Data For Predict..........")
        if input_data is None and config.input_data is not None:
            input_data = config.input_data

        if input_data is None:
            input_data = build_dataset_loader(config.eval_dataset.data_loader)

        candidate_labels = kwargs.pop("candidate_labels", None)
        if candidate_labels is None:
            if hasattr(input_data, "label_names"):
                candidate_labels = input_data.label_names
            else:
                candidate_labels = ["sunflower", "tree", "dog", "cat", "toy"]

        hypothesis_template = kwargs.pop("hypothesis_template", None)
        if hypothesis_template is None:
            if config.eval_dataset.data_loader.hypothesis_template is not None:
                hypothesis_template = config.eval_dataset.data_loader.hypothesis_template
            else:
                hypothesis_template = "{}"

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='zero_shot_image_classification',
                                    network=network,
                                    tokenizer=tokenizer,
                                    image_processor=image_processor,
                                    candidate_labels=candidate_labels,
                                    hypothesis_template=hypothesis_template,
                                    **kwargs)
