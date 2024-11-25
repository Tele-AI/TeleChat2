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
"""Contrastive Language Image Pretrain Trainer."""
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
class ContrastiveLanguageImagePretrainTrainer(BaseTrainer):
    """
    Contrastive Language Image Pretrain Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None

    Raises:
        NotImplementedError: If evaluate method or predict method not implemented.

    Examples:
        >>> from mindformers import ContrastiveLanguageImagePretrainTrainer
        >>> trainer = ContrastiveLanguageImagePretrainTrainer(model_name="clip_vit_b_b32")
        >>> type(trainer)
        <class 'mindformers.trainer.contrastive_language_image_pretrain.
        contrastive_language_image_pretrain.ContrastiveLanguageImagePretrainTrainer'>
    """

    def __init__(self, model_name: str = None):
        super(ContrastiveLanguageImagePretrainTrainer, self).__init__(
            "contrastive_language_image_pretrain", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              optimizer: Optional[Optimizer] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Train For Trainer.

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

    def evaluate(self, **kwargs):
        raise NotImplementedError(
            "The contrastive language image pretrain task does not support evaluate.")

    def predict(self, **kwargs):
        raise NotImplementedError(
            "The contrastive language image pretrain task does not support predict.")
