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
"""utils"""

import mindspore as ms
from mindspore.communication.management import init

from mindformers.tools.logger import logger
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_world_size, initialize_model_parallel


def decay_filter(x):
    return "norm" not in x.name.lower() and "bias" not in x.name.lower()


def set_weight_decay(params, weight_decay=1e-1):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest

    Args:
        params (list[Parameter]): List of parameters to apply weight decay to.

    Returns:
        list: A list of dictionaries specifying the parameter groups and their respective weight decay coefficients.
    """
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = []
    if decay_params:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})
    if other_params:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    return group_params


def set_parallel_context(parallel_config):
    """
    Sets the parallel context based on the provided parallel configuration.

    Args:
        parallel_config (MindFormerConfig): The parallel configuration object containing the parallel settings.

    Returns:
        ParallelConfig: The updated parallel configuration object.

    """
    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
    )
    logger.warning(
        f"dp {get_data_parallel_world_size()} | pp {parallel_config.pipeline_model_parallel_size} | "
        + f"tp {parallel_config.tensor_model_parallel_size} | sp {parallel_config.sequence_parallel} | "
        + f"vpp {parallel_config.virtual_pipeline_model_parallel_size}"
    )


def set_seed(seed):
    """
    Set the seed for random number generation.

    Parameters:
    - seed (int): The seed value to set.

    Returns:
    None
    """
    # set global seed, np seed, and dataset seed
    ms.set_seed(seed)
    # set rng seed
    ms.manual_seed(seed)
