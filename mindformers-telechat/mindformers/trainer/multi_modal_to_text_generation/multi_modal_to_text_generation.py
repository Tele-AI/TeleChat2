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
"""MultiModalToTextGeneration Trainer."""
from typing import Optional, Union, List

from mindspore import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.nn import Cell, Optimizer, TrainOneStepCell

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, BaseProcessor, build_processor
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from ...dataset.dataloader import build_dataset_loader
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer

__all__ = ['MultiModalToTextGenerationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class MultiModalToTextGenerationTrainer(BaseTrainer):
    """
    MultiModalToTextGenerationTrainer Task For Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None

    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(MultiModalToTextGenerationTrainer, self).__init__("multi_modal_to_text_generation", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              optimizer: Optional[Optimizer] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Train For MultiModalToTextGenerationTrainer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer.It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]):
                The training dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]):
                The optimizer used for training. Default: None.
            wrapper (Optional[TrainOneStepCell]):
                Wraps the `network` with the `optimizer`. It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The training callback function. It supports CallBack or CallBack List of MindSpore. Default: None.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            optimizer=optimizer,
            wrapper=wrapper,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The MultiModalToTextGenerationTrainer task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[GeneratorDataset, list]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                processor: Optional[BaseProcessor] = None, **kwargs):
        """
        Predict task for MultiModalToTextGenerationTrainer Trainer.
        This function is used to predict the network.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[GeneratorDataset, list]]):
                The dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            processor (Optional[BaseProcessor]):
                Used for image and text process. Default: None.
        """
        config = self.set_config(config)

        logger.info(".........Build Input Data For Predict..........")

        if input_data is None:
            input_data = build_dataset_loader(config.eval_dataset.data_loader)

        if processor is None:
            processor = build_processor(config.processor)

        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is not None:
            processor.tokenizer = tokenizer
        else:
            tokenizer = processor.tokenizer

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='multi_modal_to_text_generation',
                                    network=network,
                                    processor=processor,
                                    tokenizer=tokenizer,
                                    **kwargs)
