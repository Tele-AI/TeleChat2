# Copyright 2024-2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auto class initialization."""

from .configuration_auto import (
    AutoConfig,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES
)
from .modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskGeneration,
    AutoModelForMaskedImageModeling,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelWithLMHead,
    MODEL_FOR_CAUSAL_LM_MAPPING
)
from .image_processing_auto import (
    AutoImageProcessor,
    IMAGE_PROCESSOR_MAPPING,
    ImageProcessingMixin
)
from .tokenization_auto import (
    AutoTokenizer,
    TOKENIZER_MAPPING
)
from .processing_auto import AutoProcessor

__all__ = [
    'AutoConfig', 'AutoModel', 'AutoModelForCausalLM', 'AutoProcessor',
    'AutoTokenizer',
]
