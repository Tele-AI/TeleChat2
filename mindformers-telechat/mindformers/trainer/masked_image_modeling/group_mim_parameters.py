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
# This file was refer to project:
# https://github.com/microsoft/SimMIM
# ============================================================================
"""group parameters for masked image modeling."""
from mindformers.tools.logger import logger


def get_group_parameters(config, model):
    """get pretrain param groups"""
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info('No weight decay: %s', skip)
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info('No weight decay keywords: %s', skip_keywords)

    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for param in model.trainable_params():
        if len(param.shape) == 1 or param.name.endswith(".bias") or (param.name in skip) or \
                check_keywords_in_name(param.name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(param.name)
        else:
            has_decay.append(param)
            has_decay_name.append(param.name)
    logger.info('No decay params: %s', no_decay_name)
    logger.info('Has decay params: %s', has_decay_name)
    return [{'params': has_decay, 'weight_decay': config.optimizer.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
            {'order_params': model.trainable_params()}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
            break
    return isin
