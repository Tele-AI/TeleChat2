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
"""
Test module for testing the interface used for mindformers.
How to run this:
python tests/ut/test_build.py
"""
import os
from dataclasses import dataclass
from typing import Callable
import pytest
from mindspore.nn import AdamWeightDecay, CosineDecayLR, Accuracy,\
    TrainOneStepWithLossScaleCell, L1Loss
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore import Parameter, Tensor

from mindformers.tools.logger import logger
from mindformers.tools import MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers.core import build_lr, build_optim, build_loss, build_metric
from mindformers.trainer import build_trainer, BaseTrainer
from mindformers.models import build_model, build_processor, build_network
from mindformers.models import PretrainedConfig, PreTrainedModel
from mindformers.dataset import build_dataset, build_sampler, check_dataset_config, \
    build_dataset_loader, build_mask, build_transforms, BaseDataset
from mindformers.pipeline import build_pipeline
from mindformers.wrapper import build_wrapper

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

yaml_path = os.path.join(path, 'tests', 'st', 'test_build.yaml')
all_config = MindFormerConfig(yaml_path)


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class TestDataLoader:
    """Test DataLoader API For Register."""
    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir


@MindFormerRegister.register(MindFormerModuleType.DATASET_SAMPLER)
class TestSampler:
    """Test Sampler API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TestTransforms1:
    """Test Transforms API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TestTransforms2:
    """Test Transforms API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class TestModelMask:
    """Test Model Mask API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class TestAttentionModule:
    """Test Module API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TestDataset(BaseDataset):
    """Test Dataset API For Register."""
    def __init__(self, dataset_config: dict = None):
        super(TestDataset, self).__init__(dataset_config)
        self.config = dataset_config

    def __new__(cls, dataset_config: dict = None):
        if dataset_config is not None:
            build_dataset_loader(dataset_config.data_loader)
            logger.info("Test Build DataLoader Success")
            build_sampler(dataset_config.sampler)
            logger.info("Test Build Sampler Success")
            build_transforms(dataset_config.transforms)
            logger.info("Test Build Transforms Success")
            build_mask(dataset_config.mask_policy)
            logger.info("Test Build Mask Policy Success")
        else:
            build_dataset_loader(class_name='TestDataLoader')
            logger.info("Test Build DataLoader Success")
            build_sampler(class_name='TestSampler')
            logger.info("Test Build Sampler Success")
            build_transforms(class_name='TestTransforms1')
            logger.info("Test Build Transforms Success")
            build_mask(class_name='TestModelMask')
            logger.info("Test Build Mask Policy Success")


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestTextConfig(PretrainedConfig):
    """Test TextConfig API For Register."""
    seq_length: int = 12


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestVisionConfig(PretrainedConfig):
    """Test VisionConfig API For Register."""
    seq_length: int = 12


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestModelConfig(PretrainedConfig):
    """Test ModelConfig API For Register."""
    batch_size: int = 2
    embed_dim: int = 768
    text_config: Callable = TestTextConfig
    vision_config: Callable = TestVisionConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class TestModel(PreTrainedModel):
    """Test Model API For Register."""
    def __init__(self, config: PretrainedConfig = None):
        super(TestModel, self).__init__(config)
        self.model_config = config
        self.params = Parameter(Tensor([0.1]))


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class TestTokenizer:
    """Test Tokenizer API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class TestAdamWeightDecay(AdamWeightDecay):
    """Test AdamWeightDecay API For Register."""
    def __init__(self, params, learning_rate=1e-3,
                 beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(TestAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                  beta1=beta1, beta2=beta2, eps=eps,
                                                  weight_decay=weight_decay)
        self.param = params


@MindFormerRegister.register(MindFormerModuleType.LR)
class TestCosineDecayLR(CosineDecayLR):
    """Test CosineDecayLR API For Register."""
    def __init__(self, min_lr, max_lr, decay_steps):
        super(TestCosineDecayLR, self).__init__(min_lr, max_lr, decay_steps)
        self.lr = max_lr


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class TestTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """Test TrainOneStepWithLossScaleCell API For Register."""
    def __init__(self, network, optimizer, scale_sense):
        super(TestTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_sense)
        self.scale_sense = scale_sense


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class TestAccuracy(Accuracy):
    """Test Accuracy API For Register."""
    def __init__(self, eval_type='classification'):
        super(TestAccuracy, self).__init__(eval_type)
        self.eval = eval_type


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class TestL1Loss(L1Loss):
    """Test L1Loss API For Register."""
    def __init__(self, reduction='mean'):
        super(TestL1Loss, self).__init__(reduction)
        self.reduction = reduction


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TestLLossMonitor(LossMonitor):
    """Test LossMonitor API For Register."""
    def __init__(self, per_print_times=1):
        super(TestLLossMonitor, self).__init__(per_print_times)
        self.print = per_print_times


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TestTimeMonitor(TimeMonitor):
    """Test TimeMonitor API For Register."""
    def __init__(self, data_size=1):
        super(TestTimeMonitor, self).__init__(data_size)
        self.data_size = data_size


@MindFormerRegister.register(MindFormerModuleType.PIPELINE)
class TestPipeline:
    """Test Pipeline API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class TestProcessor:
    """Test Processor API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class TestTaskTrainer(BaseTrainer):
    """Test TimeMonitor API For Register."""
    def __init__(self, model_name='vit'):
        super(TestTaskTrainer, self).__init__(model_name=model_name)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_from_config():
    """
    Feature: Build API from config
    Description: Test build function to instance API from config
    Expectation: TypeError
    """
    # build dataset
    check_dataset_config(all_config)
    build_dataset(all_config.train_dataset_task)

    logger.info("Test Build Dataset Success")

    # build model
    model = build_model(all_config.model)
    logger.info("Test Build Model Success")

    # build network
    model = build_network(all_config.model)
    logger.info("Test Build Network Success")

    # build lr
    lr = build_lr(all_config.lr_schedule)
    logger.info("Test Build LR Success")

    # build optimizer
    if lr is not None:
        optimizer = build_optim(all_config.optimizer,
                                default_args={"params": model.trainable_params(),
                                              "learning_rate": lr})
    else:
        optimizer = build_optim(all_config.optimizer,
                                default_args={"params": model.trainable_params()})
    logger.info("Test Build Optimizer Success")

    # build wrapper
    build_wrapper(all_config.runner_wrapper,
                  default_args={"network": model, "optimizer": optimizer})
    logger.info("Test Build Wrapper Success")

    build_loss(all_config.loss)
    logger.info("Test Build Loss Success")

    build_metric(all_config.metric)
    logger.info("Test Build Metric Success")

    build_processor(all_config.processor)
    logger.info("Test Build Processor Success")

    # build trainer
    build_trainer(all_config.trainer)
    logger.info("Test Build Trainer Success")

    # build pipeline
    build_pipeline(all_config.pipeline)
    logger.info("Test Build Pipeline Success")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_from_class_name():
    """
    Feature: Build API from class name
    Description: Test build function to instance API from class name
    Expectation: TypeError
    """
    # build dataset
    build_dataset(class_name='TestDataset')
    logger.info("Test Build Dataset Success")
    # build model
    model = build_model(class_name='TestModel', config=TestModelConfig())
    logger.info("Test Build Network Success")
    # build lr
    lr = build_lr(class_name='TestCosineDecayLR', min_lr=0., max_lr=0.001, decay_steps=1000)
    logger.info("Test Build LR Success")
    # build optimizer
    if lr is not None:
        optimizer = build_optim(class_name='TestAdamWeightDecay',
                                params=model.trainable_params(), learning_rate=lr)
    else:
        optimizer = build_optim(class_name='TestAdamWeightDecay', params=model.trainable_params())
    logger.info("Test Build Optimizer Success")

    # build loss
    build_loss(class_name='TestL1Loss')
    logger.info("Test Build Loss Success")

    # build metric
    build_metric(class_name='TestAccuracy')
    logger.info("Test Build Metric Success")

    # build wrapper
    scale_sense = Tensor(1.0)
    build_wrapper(class_name='TestTrainOneStepWithLossScaleCell',
                  network=model, optimizer=optimizer, scale_sense=scale_sense)
    logger.info("Test Build Wrapper Success")

    # build processor
    build_processor(class_name='TestProcessor')
    logger.info("Test Build Processor Success")

    # build trainer
    build_trainer(class_name='TestTaskTrainer')
    logger.info("Test Build Trainer Success")

    # build pipeline
    build_pipeline(class_name='TestPipeline')
    logger.info("Test Build Pipeline Success")
