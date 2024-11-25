# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Token Classification Trainer."""
from typing import Optional, List, Union
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.nn import TrainOneStepCell, Optimizer, Cell

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ..base_trainer import BaseTrainer
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ...dataset.labels import cluener_labels


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class TokenClassificationTrainer(BaseTrainer):
    """
    Trainer of token classification task. It provides training, evaluation and prediction interfaces for
    question answering task, allowing users to quickly start the process according to the model name,
    and also provides a large number of customizable items to meet user needs.

    Args:
        model_name (str): The model name of token classification task trainer. Default: None

    Raises:
        NotImplementedError: If train method, evaluate method or predict method not implemented.
    """

    def __init__(self, model_name: str = None):
        super(TokenClassificationTrainer, self).__init__("token_classification", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        The training API of token classification task. It allows to quickly start training or fine-tuning based on
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

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                 network: Optional[Union[Cell, PreTrainedModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """
        The evaluation API of token classification task. It allows to quickly start evaluation based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        dataset, callbacks, compute_metrics and callbacks.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset]]): The evaluate dataset.
                It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The eval callback function.
                It supports CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It supports dict or set in MindSpore's Metric class.
                Default: None.

        Returns:
            None
        """
        metric_name = "Entity Metric"
        kwargs.setdefault("metric_name", metric_name)
        super().evaluate_process(
            config=config,
            network=network,
            dataset=dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                **kwargs):
        """
        The prediction API of token classification task. It allows to quickly start prediction based on
        initialization conditions or by passing in custom configurations. The configurable items include the network,
        input data, and tokenizer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]): The predict data. Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer for tokenizing the input text.
                Default: None.

        Returns:
            A list of prediction results.
        """
        config = self.set_config(config)

        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list)):
            raise ValueError("Input data's type must be one of [str, list]")

        if isinstance(input_data, list):
            for item in input_data:
                if not isinstance(item, str):
                    raise ValueError("The element of input data list must be str")

        # This is a known issue, you need to specify batch size equal to 1 when creating bert model.
        config.model.model_config.batch_size = 1

        max_length = network.config.seq_length if network else config.model.model_config.seq_length

        id2label = {label_id: label for label_id, label in enumerate(cluener_labels)}

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='token_classification',
                                    network=network,
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    padding="max_length",
                                    id2label=id2label,
                                    **kwargs)
