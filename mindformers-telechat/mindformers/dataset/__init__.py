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
"""MindFormers Dataset."""
from .dataloader import (
    ADGenDataLoader,
    BaseMultiModalDataLoader,
    CLUENERDataLoader,
    Cifar100DataLoader,
    CommonDataLoader,
    Flickr8kDataLoader,
    IndexedDataLoader,
    MultiImgCapDataLoader,
    MultiSourceDataLoader,
    SFTDataLoader,
    SQuADDataLoader,
    ToolAlpacaDataLoader,
    TrainingDataLoader,
    WMT16DataLoader
)
from .mask import (
    MaeMask,
    SimMask
)
from .transforms import (
    BCHW2BHWC,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    CaptionTransform,
    LabelPadding,
    Mixup,
    RandomChoiceTokenizerForward,
    RandomCropDecodeResize,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    TokenizeWithLabel,
    TokenizerForward,
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform
)
from .handler import (
    AdgenInstructDataHandler,
    AlpacaInstructDataHandler,
    CodeAlpacaInstructDataHandler,
    LlavaInstructDataHandler,
    build_data_handler
)
from .build_dataset import build_dataset
from .dataloader.build_dataloader import build_dataset_loader
from .mask.build_mask import build_mask
from .sampler.build_sampler import build_sampler
from .transforms.build_transforms import build_transforms
from .base_dataset import BaseDataset
from .causal_language_model_dataset import CausalLanguageModelDataset
from .contrastive_language_image_pretrain_dataset import ContrastiveLanguageImagePretrainDataset
from .img_cls_dataset import ImageCLSDataset
from .keyword_gen_dataset import KeyWordGenDataset
from .mask_language_model_dataset import MaskLanguageModelDataset
from .mim_dataset import MIMDataset
from .question_answering_dataset import QuestionAnsweringDataset
from .reward_model_dataset import RewardModelDataset
from .text_classification_dataset import TextClassificationDataset
from .token_classification_dataset import TokenClassificationDataset
from .translation_dataset import TranslationDataset
from .zero_shot_image_classification_dataset import ZeroShotImageClassificationDataset
from .multi_turn_dataset import MultiTurnDataset
from .general_dataset import GeneralDataset
from .utils import (
    check_dataset_config,
    check_dataset_iterable
)
from .modal_to_text_sft_dataset import ModalToTextSFTDataset

__all__ = [
    'CausalLanguageModelDataset', 'ContrastiveLanguageImagePretrainDataset', 'KeyWordGenDataset', 'MultiTurnDataset',
]
