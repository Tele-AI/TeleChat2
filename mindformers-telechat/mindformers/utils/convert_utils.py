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
Convert utils.
"""
import torch
import mindspore as ms
from mindspore.ops.operations import Cast

cpu_cast = Cast().set_device("CPU")


def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    if value.dtype == torch.bfloat16:
        np_value = value.detach().cpu().to(torch.float32).numpy()
    else:
        np_value = value.detach().numpy()

    if dtype:
        return ms.Tensor(np_value, dtype=dtype)
    return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)


def ms2pt(value: ms.Tensor, dtype) -> torch.Tensor:
    """
    convert ms.Tensor to torch.Tensor with specified dtype
    """
    if value.dtype == ms.bfloat16:
        np_value = cpu_cast(value, ms.float32).asnumpy()
    else:
        np_value = value.asnumpy()

    if dtype:
        return torch.from_numpy(np_value).cpu().to(dtype)
    return torch.from_numpy(np_value).cpu().to(torch.bfloat16) if value.dtype == ms.bfloat16 else torch.from_numpy(
        np_value)


def is_lora_param(key: str) -> bool:
    """
    is lora parameter of model weight
    """
    if 'lora' in key.lower():
        return True
    return False
