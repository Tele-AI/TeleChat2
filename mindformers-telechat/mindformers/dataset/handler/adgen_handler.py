# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Adgen Dataset Handler."""
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.dataset.handler.base_handler import BaseInstructDataHandler


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AdgenInstructDataHandler(BaseInstructDataHandler):
    """adgen data handler"""
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content": "prompt", "summary": "answer"})

    def format_func(self, example):
        logger.info(f"nothing to do")
