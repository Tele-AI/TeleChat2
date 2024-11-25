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
"""Masked Image Modeling Trainer."""
from typing import Optional, List, Union

import numpy as np
from PIL.Image import Image
from mindspore import Tensor
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from mindformers.trainer.config_args import ConfigArguments
from mindformers.trainer.training_args import TrainingArguments
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.tools.logger import logger
from mindformers.tools.image_tools import load_image


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class MaskedImageModelingTrainer(BaseTrainer):
    """
    MaskedImageModeling Task For Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None

    Raises:
        NotImplementedError: If evaluate method or predict method not implemented.

    Examples:
        >>> from mindformers.trainer import MaskedImageModelingTrainer
        >>> trainer = MaskedImageModelingTrainer(model_name="mae_vit_base_p16")
        >>> type(trainer)
        <class 'mindformers.trainer.masked_image_modeling.masked_image_modeling_pretrain.MaskedImageModelingTrainer'>
    """
    def __init__(self, model_name: str = None):
        super(MaskedImageModelingTrainer, self).__init__("masked_image_modeling", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Train task for MaskedImageModeling Trainer.
        This function is used to train or fine-tune the network.
        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]):
                The training dataset. It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]):
                The training network's optimizer. It support Optimizer class of MindSpore. Default: None.
            wrapper (Optional[TrainOneStepCell]):
                Wraps the `network` with the `optimizer`. It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The training callback function. It support CallBack or CallBack List of MindSpore.
                Default: None.
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
            "The MaskedImageModeling task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """
        Predict task for MaskedImageModeling Trainer.
        This function is used to predict the network.
        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, tokenizer, image_processor, audio_processor.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, np.ndarray, Image, str, list]]):
                The predict data. Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            image_processor (Optional[BaseImageProcessor]):
                The processor for image preprocessing. It support BaseImageProcessor class.
                Default: None.

        Returns:
            List, a list of prediction.
        """
        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = config.input_data
        if not isinstance(input_data, (Tensor, np.ndarray, Image, str, list)):
            raise ValueError("Input data's type must be one of "
                             "[str, ms.Tensor, np.ndarray, PIL.Image.Image, list]")
        batch_input_data = []
        if isinstance(input_data, str):
            batch_input_data.append(load_image(input_data))
        elif isinstance(input_data, list):
            for data_path in input_data:
                batch_input_data.append(load_image(data_path))
        else:
            batch_input_data = input_data

        return self.predict_process(config=config,
                                    input_data=batch_input_data,
                                    task='masked_image_modeling',
                                    network=network,
                                    image_processor=image_processor,
                                    **kwargs)
