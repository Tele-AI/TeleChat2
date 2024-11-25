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
# ============================================================================
"""Grouped parameters of optimizer."""

import json
from functools import partial
from typing import Optional

from mindspore.nn import Cell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindformers.models import PreTrainedModel
from mindformers.core.lr import LearningRateWiseLayer
from mindformers.tools.logger import logger
from .utils import check_keywords_in_name


def get_optimizer_grouped_parameters(model: Optional[PreTrainedModel] = None,
                                     weight_decay: float = 0.0,
                                     dynamic_lr_schedule: Optional[LearningRateSchedule] = None,
                                     layer_scale: bool = False, layer_decay: float = 1.0):
    """Get grouped parameters of the network for training."""
    if not isinstance(model, (Cell, PreTrainedModel)):
        raise TypeError(f"model type should be PreTrainedModel, but get {type(model)}")

    skip_params = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_params = model.no_weight_decay()
        logger.info('No weight decay: %s', skip_params)
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info('No weight decay keywords: %s', skip_keywords)

    decay_parameters_names = []
    for param in model.trainable_params():
        if skip_params or skip_keywords:
            if param.name in skip_params:
                continue
            if check_keywords_in_name(param.name, skip_keywords):
                continue
        elif len(param.shape) == 1 or param.name.endswith(".bias"):
            continue
        decay_parameters_names.append(param.name)

    get_layer_id_func = None
    scales_list = []
    if dynamic_lr_schedule is not None:
        if layer_scale:
            if model.__class__.__name__ == 'SwinForImageClassification':
                depths = model.config.depths
                num_layers = sum(depths)
                get_layer_id_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
                scales_list = list(layer_decay ** i for i in reversed(range(num_layers + 2)))
            elif model.__class__.__name__ == 'ViTForImageClassification':
                num_layers = model.config.depth
                get_layer_id_func = partial(get_vit_layer, num_layers=num_layers + 2)
                scales_list = list(layer_decay ** i for i in reversed(range(num_layers + 2)))
            else:
                logger.warning("Now only support VitModel and SwinModel,"
                               "if use dynamic_lr_schedule and layer_scale, they will be reset and invalid.")
                layer_scale = False
                dynamic_lr_schedule = None
        else:
            logger.warning("dynamic_lr_schedule will be reset and invalid when layer_scale is False.")
            dynamic_lr_schedule = None

    parameter_group_names = {}
    parameter_group_vars = {}
    scale = 1.
    for param in model.trainable_params():
        if param.name in decay_parameters_names:
            group_name = 'decay'
        else:
            group_name = 'no_decay'
            weight_decay = 0.

        if get_layer_id_func is not None:
            layer_id = get_layer_id_func(param.name)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if scales_list is not None:
                scale = scales_list[layer_id]

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
            }
            if isinstance(dynamic_lr_schedule, LearningRateSchedule):
                if layer_scale:
                    parameter_group_vars[group_name]["lr"] = LearningRateWiseLayer(dynamic_lr_schedule, scale)
                else:
                    parameter_group_vars[group_name]["lr"] = dynamic_lr_schedule

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
        layer_id = int(name.split('.')[2])
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
