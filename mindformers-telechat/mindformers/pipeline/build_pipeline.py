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
"""Build Pipeline API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_pipeline(
        config: dict = None, default_args: dict = None,
        module_type: str = 'pipeline', class_name: str = None, **kwargs):
    r"""Build Pipeline API.
    Instantiate the task pipeline from MindFormerRegister's registry.

    Args:
        config (dict):
            The task pipeline's config.
        default_args (dict):
            The default args of pipeline.
        module_type (str):
            The module type of MindFormerModuleType. Default: 'pipline'.
        class_name (str):
            The class name of task pipeline.

    Returns:
        The task pipeline instance by config.

    Examples:
        >>> from mindformers import build_pipeline
        >>> pipeline_config = {'type': 'zero_shot_image_classification',
            'model': 'clip_vit_b_32',
            'candidate_labels': ["sunflower", "tree", "dog", "cat", "toy"],
            'hypothesis_template': "This is a photo of {}."}
        >>> classifier = build_pipeline(pipeline_config)
        >>> type(classifier)
            <class 'mindformers.pipeline.zero_shot_image
            _classification_pipeline.ZeroShotImageClassificationPipeline'>
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.PIPELINE, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
