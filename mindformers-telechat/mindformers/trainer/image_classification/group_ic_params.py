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
import json
from functools import partial

from mindformers.core.lr import LearningRateWiseLayer
from mindformers.tools.logger import logger


def get_group_parameters(config, model, base_lr):
    """get finetune param groups"""
    if config.model.arch.type == 'SwinForImageClassification':
        depths = config.model.model_config.depths
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    elif config.model.arch.type == 'ViTForImageClassification':
        num_layers = config.model.model_config.num_hidden_layers
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    else:
        raise NotImplementedError

    layer_decay = config.runner_config.layer_decay
    scales = list(layer_decay ** i for i in reversed(range(num_layers + 2)))

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info('No weight decay: %s', skip)
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info('No weight decay keywords: %s', skip_keywords)

    parameter_group_names = {}
    parameter_group_vars = {}

    for param in model.trainable_params():

        if len(param.shape) == 1 or param.name.endswith(".bias") or (param.name in skip) or \
                check_keywords_in_name(param.name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = config.optimizer.weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(param.name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": LearningRateWiseLayer(base_lr, scale),
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(param.name)
    param_groups = json.dumps(parameter_group_names, indent=2)
    logger.info("Param groups = %s", param_groups)
    return list(parameter_group_vars.values())


def get_vit_layer(name, num_layers):
    """get vit layer"""
    if name.endswith(("cls_tokens", "mask_tokens", "pos_embed")):
        layer_num = 0
    elif name.startswith("vit.patch_embed"):
        layer_num = 0
    elif name.startswith("vit.blocks"):
        layer_id = int(name.split('.')[1])
        layer_num = layer_id + 1
    else:
        layer_num = num_layers - 1
    return layer_num


def get_swin_layer(name, num_layers, depths):
    """get swin layer"""
    if name in ("encoder.mask_token",):
        layer_num = 0
    elif name.startswith("encoder.patch_embed"):
        layer_num = 0
    elif name.startswith("encoder.layers"):
        layer_id = int(name.split('.')[2])
        block_id = name.split('.')[4]
        if block_id in ('reduction', 'norm'):
            layer_num = sum(depths[:layer_id + 1])
        else:
            layer_id = sum(depths[:layer_id]) + int(block_id)
            layer_num = layer_id + 1
    else:
        layer_num = num_layers - 1
    return layer_num


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
            break
    return isin
