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
"""Build Trainer API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from . import ImageClassificationTrainer, ZeroShotImageClassificationTrainer, \
    MaskedImageModelingTrainer, MaskedLanguageModelingTrainer, ImageToTextRetrievalTrainer, \
    TranslationTrainer, TokenClassificationTrainer, TextClassificationTrainer, \
    ContrastiveLanguageImagePretrainTrainer, QuestionAnsweringTrainer, GeneralTaskTrainer


def build_trainer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'trainer', class_name: str = None, **kwargs):
    r"""Build trainer API.
    Instantiate the task trainer from MindFormerRegister's registry.

    Args:
        config (dict): The task trainer's config. Default: None.
        default_args (dict): The default argument of trainer API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'trainer'.
        class_name (str): The class name of task trainer API. Default: None.

    Return:
        The function instance of task trainer API.

    Examples:
        >>> from mindformers import build_trainer
        >>> trainer_config = {'type': 'image_classification', 'model_name': 'vit'}
        >>> # 1) use config dict to build trainer
        >>> cls_trainer_config = build_trainer(trainer_config)
        >>> # 2) use class name to build trainer
        >>> cls_trainer_class_name = build_trainer(class_name='image_classification', model_name='vit')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.TRAINER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_mf_trainer():
    """ register MindFomrers builtin LR class. """
    # adapt huggingface
    MindFormerRegister.register_cls(
        ImageClassificationTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="image_classification")

    MindFormerRegister.register_cls(
        ZeroShotImageClassificationTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="zero_shot_image_classification")

    MindFormerRegister.register_cls(
        MaskedImageModelingTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="masked_image_modeling")

    MindFormerRegister.register_cls(
        MaskedLanguageModelingTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="fill_mask")

    MindFormerRegister.register_cls(
        TokenClassificationTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="token_classification")

    MindFormerRegister.register_cls(
        TextClassificationTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="text_classification")

    MindFormerRegister.register_cls(
        ContrastiveLanguageImagePretrainTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="contrastive_language_image_pretrain")

    MindFormerRegister.register_cls(
        TranslationTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="translation")

    MindFormerRegister.register_cls(
        QuestionAnsweringTrainer,
        module_type=MindFormerModuleType.TRAINER, alias="question_answering")

    MindFormerRegister.register_cls(
        ImageToTextRetrievalTrainer, module_type=MindFormerModuleType.TRAINER, alias="image_to_text_retrieval")

    MindFormerRegister.register_cls(
        GeneralTaskTrainer, module_type=MindFormerModuleType.TRAINER, alias="general")


register_mf_trainer()
