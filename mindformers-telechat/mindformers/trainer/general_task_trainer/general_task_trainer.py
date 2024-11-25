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
"""General Task Example For Trainer."""
from typing import Optional, List, Union

from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.nn import TrainOneStepCell, Optimizer, Cell

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class GeneralTaskTrainer(BaseTrainer):
    r"""General Task Example For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset import GeneratorDataset
        >>> from mindspore.nn import AdamWeightDecay, WarmUpLR, \
        ...      DynamicLossScaleUpdateCell, TrainOneStepWithLossScaleCell, Accuracy
        >>> from mindformers.trainer import GeneralTaskTrainer
        >>> from mindformers.models import ViTForImageClassification, ViTConfig, ViTImageProcessor
        >>> class MyDataLoader:
        ...    def __init__(self):
        ...        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]
        ...        self._label = [np.ones(1000, np.float32) for _ in range(64)]
        ...
        ...    def __getitem__(self, index):
        ...        return self._data[index], self._label[index]
        ...
        ...    def __len__(self):
        ...        return len(self._data)
        >>> train_dataset = GeneratorDataset(source=MyDataLoader(), column_names=['image', 'label'])
        >>> train_dataset = train_dataset.batch(batch_size=2)
        >>> eval_dataset = GeneratorDataset(source=MyDataLoader(), column_names=['image', 'label'])
        >>> eval_dataset = eval_dataset.batch(batch_size=2)
        >>> general_task = GeneralTaskTrainer(model_name='common')
        >>> vit_config = ViTConfig(batch_size=2)
        >>> network_with_loss = ViTForImageClassification(vit_config)
        >>> lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
        >>> optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
        ...                             learning_rate=lr_schedule,
        ...                             params=network_with_loss.trainable_params())
        >>> loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> wrapper = TrainOneStepWithLossScaleCell(network_with_loss, optimizer, scale_sense=loss_scale)
        >>> general_task.train(wrapper=wrapper, dataset=train_dataset)
        >>> compute_metrics = {"Accuracy": Accuracy(eval_type='classification')}
        >>> general_task.evaluate(network=network_with_loss, dataset=eval_dataset, compute_metrics=compute_metrics)
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(GeneralTaskTrainer, self).__init__("general", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for General-Trainer.
        This function is used to train or fine-tune the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The training dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
                Default: None.
            wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
                It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.

        Raises:
            NotImplementedError: If network or dataset not implemented.
        """
        if network is None:
            raise NotImplementedError("train network must be define, but get None.")

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
        r"""Evaluate task for General-Trainer.
        This function is used to evaluate the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, dataset, callback, compute_metrics.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]): The network for trainer.
                It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The evaluate dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.
            compute_metrics (Optional[Union[dict, set]]): The metric of evaluating.
                It support dict or set in MindSpore's Metric class.
                Default: None.

        Raises:
            NotImplementedError: If network or dataset or compute_metrics not implemented.
        """
        if network is None:
            raise NotImplementedError("eval network must be define, but get None.")

        metric_name = "General Task Metrics"
        kwargs.setdefault("metric_name", metric_name)
        self.evaluate_process(
            config=config,
            network=network,
            dataset=dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self, **kwargs):
        raise NotImplementedError(
            "The general task does not support predict, please customize pipeline inference.")
