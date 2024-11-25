# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023-2024 Huawei Technologies Co., Ltd
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

"""Constant Declaration of Pipeline Registry"""
from mindformers.models.auto import AutoModel
from mindformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskGeneration,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForZeroShotImageClassification)

from .fill_mask_pipeline import FillMaskPipeline
from .image_classification_pipeline import ImageClassificationPipeline
from .image_to_text_generation_pipeline import ImageToTextPipeline
from .masked_image_modeling_pipeline import MaskedImageModelingPipeline
from .pipeline_registry import PipelineRegistry
from .question_answering_pipeline import QuestionAnsweringPipeline
from .segment_anything_pipeline import SegmentAnythingPipeline
from .text_classification_pipeline import TextClassificationPipeline
from .text_generation_pipeline import TextGenerationPipeline
from .token_classification_pipeline import TokenClassificationPipeline
from .translation_pipeline import TranslationPipeline
from .zero_shot_image_classification_pipeline import \
    ZeroShotImageClassificationPipeline

TASK_ALIASES = {
    "text_classification": "text-classification",
    "sentiment_analysis": "text-classification",
    "ner": "token-classification",
    "fill_mask": "fill-mask",
    "image_classification": "image-classification",
    "image_to_text_generation": "image-to-text",
    "masked_image_modeling": "masked-image-modeling",
    "question_answering": "question-answering",
    "segment_anything": "segment-anything",
    "text_generation": "text-generation",
    "token_classification": "token-classification",
    "zero_shot_image_classification": "zero-shot-image-classification",
}

# default model repo need to be filled.
SUPPORTED_TASKS = {
    # fill mask model
    "fill-mask": {
        "impl": FillMaskPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "ms": (AutoModelForImageClassification,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "image-to-text": {
        "impl": ImageToTextPipeline,
        "ms": (AutoModelForVision2Seq,),
        "default": {"model": {"ms": ()}},
        "type": "multimodal",
    },
    "masked-image-modeling": {
        "impl": MaskedImageModelingPipeline,
        "ms": (AutoModelForMaskedImageModeling,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "ms": (AutoModelForQuestionAnswering,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "segment-anything": {
        "impl": SegmentAnythingPipeline,
        "ms": (AutoModelForMaskGeneration,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "ms": (AutoModelForSequenceClassification,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "ms": (AutoModelForCausalLM,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "ms": (AutoModelForTokenClassification,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "translation": {
        "impl": TranslationPipeline,
        "ms": (AutoModelForSeq2SeqLM,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "zero-shot-image-classification": {
        "impl": ZeroShotImageClassificationPipeline,
        "ms": (AutoModelForZeroShotImageClassification,),
        "default": {"model": {"ms": ()}},
        "type": "multimodal",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_IMAGE_PROCESSOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
# Those model configs are special, they are generic over their task, meaning
# any tokenizer/feature_extractor might be used for a given model, so we cannot
# use the statically defined TOKENIZER_MAPPING and FEATURE_EXTRACTOR_MAPPING to
# see if the model defines such objects or not.
# reserved multi-modal configs
MULTI_MODEL_CONFIGS = {}
for task, values in SUPPORTED_TASKS.items():
    if values.get("type") == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values.get("type") in {"image", "video"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values.get("type") in {"audio"}:
        NO_TOKENIZER_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values.get("type") != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values.get('type')}")

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)
