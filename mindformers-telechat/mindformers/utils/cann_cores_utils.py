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

"""
Utils for cann workqueue cores
"""

import os
import psutil

from mindformers.utils.bit_array import BitArray
from mindformers.tools.logger import logger


def get_cann_workqueue_cores(device_id: int) -> list:
    """
    get cann workqueue binding cores list
    for most system, the config is set on path:
    /sys/devices/virtual/workqueue/dev0_sq_send_wq/cpumask

    Args:
        device_id (`int`):
            The device_id for the workqueue, most time is related to rank_ik.

    Returns:
        The marked core index list.
    """
    cann_workqueue_config_path = f"/sys/devices/virtual/workqueue/dev{device_id}_sq_send_wq/cpumask"
    if not os.path.exists(cann_workqueue_config_path):
        # no this config, return [] to disable cann binding
        return []

    f = open(cann_workqueue_config_path)
    cann_config = f.read()
    cann_config = cann_config.replace(",", "")
    cann_config = cann_config.replace("\n", "")
    mask_array = BitArray()
    mask_array.load_from_str(cann_config)
    return mask_array.get_marked_index()


def mask_to_str(mask: BitArray) -> str:
    """
    convert BitArray mask to string format with workqueue config

    Args:
        mask (`BitArray`):
            The BitArray mask to convert to string.

    Returns:
        The string followed with cann workqueue format to config.
    """
    mask_bytes = mask.to_bytes_array()
    mask_str = ""
    separete_num = 4
    i = 0
    for mask_value in mask_bytes:
        mask_str += '{:02x}'.format(mask_value)
        i += 1
        if i % separete_num == 0:
            mask_str += ","
    return mask_str


def binding_cann_workqueue(device_num: int, core_num_per_workqueue: int, separate_device_cores: bool):
    """
    binding cann workqueue cores

    Args:
        device_num (`int`):
            The total device number on the server.
        core_num_per_workqueue (`int`):
            The core number for each workqueue, the core index will alloc from end core index for each device.
        separate_device_cores (`int`):
            If separate device cores, each device workqueue binding itself cores,
            otherwise, all device workqueu binding to same cores.

    Returns:
        NA.
    """
    print(f"the cann workqueue config command list in the follow, please execute the cmd by root user!")

    total_core_num = psutil.cpu_count(logical=True)
    core_num_per_device = int(total_core_num / device_num)

    device_core_mask = BitArray(total_core_num)
    for i in range(device_num):
        cann_workqueue_config_path = f"/sys/devices/virtual/workqueue/dev{i}_sq_send_wq/cpumask"
        mask = BitArray(total_core_num)
        start_core_num = i * core_num_per_device
        end_core_num = start_core_num + core_num_per_device - 1
        for j in range(core_num_per_workqueue):
            core_index = end_core_num - j
            mask[core_index] = 1
            device_core_mask[core_index] = 1

        if separate_device_cores:
            mask_str = mask_to_str(mask)
            bind_cann_core_cmd = f"echo \"{mask_str}\" > {cann_workqueue_config_path}"
            logger.info(bind_cann_core_cmd)

    if not separate_device_cores:
        device_core_mask_str = mask_to_str(device_core_mask)

        for i in range(device_num):
            cann_workqueue_config_path = f"/sys/devices/virtual/workqueue/dev{i}_sq_send_wq/cpumask"
            bind_cann_core_cmd = f"echo \"{device_core_mask_str}\" > {cann_workqueue_config_path}"
            logger.info(bind_cann_core_cmd)
