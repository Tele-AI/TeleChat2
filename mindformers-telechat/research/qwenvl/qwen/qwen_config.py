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
"""Qwen Config API."""

from mindformers import LlamaConfig, MindFormerBook, logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['QwenConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QwenConfig(LlamaConfig):
    """
    Qwen config class.

    Returns:
        Class, QwenConfig.
    """

    _support_list = MindFormerBook.get_config_support_list()['qwen']

    def __init__(self,
                 embedding_parallel_optimizer: bool = True,
                 enable_slice_dp: bool = True,
                 enable_emb_opt: bool = False,
                 **kwargs):
        if 'num_hidden_layers' in kwargs:
            logger.warning("Argument `num_hidden_layers` is deprecated. Use `num_layers` instead.")
            if kwargs.get('num_layers', None) is None:
                num_layers = kwargs.pop('num_hidden_layers')
                kwargs['num_layers'] = num_layers

        if 'num_attention_heads' in kwargs:
            logger.warning("Argument `num_attention_heads` is deprecated. Use `num_heads` instead.")
            if kwargs.get('num_heads', None) is None:
                num_heads = kwargs.pop('num_attention_heads')
                kwargs['num_heads'] = num_heads

        super().__init__(**kwargs)
        self.embedding_parallel_optimizer = embedding_parallel_optimizer
        self.enable_slice_dp = enable_slice_dp
        self.enable_emb_opt = enable_emb_opt
