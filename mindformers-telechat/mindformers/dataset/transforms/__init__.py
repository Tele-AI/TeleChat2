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
"""MindFormers Transforms API."""
from .build_transforms import build_transforms
from .vision_transforms import (
    BCHW2BHWC,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    RandomCropDecodeResize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize
)
from .text_transforms import (
    CaptionTransform,
    LabelPadding,
    RandomChoiceTokenizerForward,
    TokenizeWithLabel,
    TokenizerForward
)
from .mixup import Mixup
from .auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform
)
from .random_erasing import RandomErasing

__all__ = []
