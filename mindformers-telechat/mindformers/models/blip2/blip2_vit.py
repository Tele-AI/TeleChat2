# Copyright 2023 Huawei Technologies Co., Ltd
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
# This file was refer to project:
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""vit models for Blip2"""
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.vit.vit import ViTModel, ViTConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTModelForBlip2(ViTModel):
    """
    ViTModel For Blip2 Models, loading a pretrained weight.
    forward will return the penultimate output.
    """
    _support_list = MindFormerBook.get_config_support_list()['vit']

    def __init__(self, config: ViTConfig):
        super(ViTModelForBlip2, self).__init__(config)
        self.load_checkpoint(config)

    def construct(self, image):
        return self.construct_without_pool(image)
