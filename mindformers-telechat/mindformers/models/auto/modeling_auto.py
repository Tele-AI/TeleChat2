# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
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
""" Auto Model class."""

import warnings
from collections import OrderedDict

from mindformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
from mindformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("bert", "BertModel"),
        ("blip2", "Blip2Model"),
        ("bloom", "BloomModel"),
        ("clip", "CLIPModel"),
        ("glm", "GLMModel"),
        ("glm2", "ChatGLM2Model"),
        ("gpt2", "GPT2Model"),
        ("llama", "LlamaModel"),
        ("mae", "ViTMAEModel"),
        ("pangualpha", "PanguAlphaModel"),
        ("sam", "SamModel"),
        ("swin", "SwinModel"),
        ("t5", "T5Model"),
        ("vit", "ViTModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("bert", "BertForPreTraining"),
        ("glm", "GLMForPreTraining"),
        ("mae", "ViTMAEForPreTraining"),
        ("blip2-stage1", "Blip2Qformer"),
        ("blip2-stage2", "Blip2Llm")
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("bert", "BertForMaskedLM"),
        ("bloom", "BloomLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("glm", "GLMChatModel"),
        ("glm2", "ChatGLM2ForConditionalGeneration"),
        ("llama", "LlamaForCausalLM"),
        ("pangualpha", "PanguAlphaHeadModel"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("bert", "BertForMaskedLM"),
        ("bloom", "BloomLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("glm", "GLMChatModel"),
        ("glm2", "ChatGLM2ForConditionalGeneration"),
        ("llama", "LlamaForCausalLM"),
        ("pangualpha", "PanguAlphaHeadModel"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("vit", "ViTForMaskedImageModeling"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("swin", "SwinForImageClassification"),
        ("vit", "ViTForImageClassification"),
        ("blip2", "Blip2Classifier")
    ]
)

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip2", "Blip2ImageToTextGeneration"),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("t5", "T5ForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("gpt2", "GPT2ForSequenceClassification"),
        ("pangualpha", "PanguAlphaPromptTextClassificationModel"),
        ("bert", "BertForMultipleChoice")
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("bert", "BertForQuestionAnswering"),
    ]
)

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip2", "Blip2ForConditionalGeneration"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("bert", "BertForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("bert", "BertForMultipleChoice"),
    ]
)

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("clip", "CLIPModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone mapping
        ("swin", "SwinBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)

MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertModel"),
    ]
)


MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)
MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)
MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING


class AutoModel(_BaseAutoModelClass):
    r"""
    This is a generic model class that will be instantiated as one of the model
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).
    """
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel, checkpoint_for_example="sam_vit_b")


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    r"""
    This is a generic model class that will be instantiated as one of the model
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).
    """
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(
    AutoModelForCausalLM,
    checkpoint_for_example="bert_tiny_uncased",
    head_doc="causal language modeling"
)


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    r"""
    This is a generic model class that will be instantiated as one of the model
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).
    """
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


AutoModelForVisualQuestionAnswering = auto_class_update(AutoModelForVisualQuestionAnswering,
                                                        head_doc="visual question answering")


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    r"""
    This is a generic model class that will be instantiated as one of the model
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).
    """
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification,
    checkpoint_for_example="clip_vit_b_16",
    head_doc="zero-shot image classification"
)


class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")


class AutoModelWithLMHead(_AutoModelWithLMHead):
    """AutoModelWithLMHead Class"""
    @classmethod
    def from_config(cls, config): # pylint: disable=W0221
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_dir, *model_args, **kwargs)
