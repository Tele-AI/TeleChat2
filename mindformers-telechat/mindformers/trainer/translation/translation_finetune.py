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
"""Translation Modeling Trainer."""
import os.path
from typing import Optional, List, Union

from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase
from mindformers.tools.register import MindFormerRegister,\
    MindFormerModuleType, MindFormerConfig
from ..base_trainer import BaseTrainer
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class TranslationTrainer(BaseTrainer):
    """
    Trainer of translation task. It provides training and prediction interfaces for
    translation task, allowing users to quickly start the process according to the model name,
    and also provides a large number of customizable items to meet user needs.

    Args:
        model_name (str): The model name of translation task trainer. Default: None

    Raises:
        NotImplementedError: If train method, evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(TranslationTrainer, self).__init__("translation", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        The training API of translation task. It allows to quickly start training or fine-tuning based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        optimizer, dataset, wrapper, and callbacks.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The training dataset.
                It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
                Default: None.
            wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
                It supports TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It supports CallBack or CallBack List of MindSpore.
                Default: None.

        Returns:
            None
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The Translation task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                **kwargs):
        """
        The prediction API of translation task. It allows to quickly start prediction based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        input data, and tokenizer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]): The predict data. It supports 1) a text string to be
                translated, 1) a file name where each line is a text to be translated  and 3) a generator dataset.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer for tokenizing the input text.
                Default: None.

        Returns:
            A list of prediction results.

        """
        if input_data is None:
            input_data = config.input_data

        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, 'r') as fp:
                input_data_list = []
                for line in fp:
                    input_data_list.append(line)
                input_data = input_data_list

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='translation',
                                    network=network,
                                    tokenizer=tokenizer,
                                    **kwargs)
