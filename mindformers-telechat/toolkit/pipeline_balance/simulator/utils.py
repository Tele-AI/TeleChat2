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
"""simulator utils"""
import time
import numpy as np
from matplotlib import colors


def format_2d_inputs(a, raw, col):
    r"""format 2d inputs"""
    if isinstance(a, (int, float)):
        return np.broadcast_to(a, (raw, col))
    if isinstance(a, (list, tuple)):
        if all(isinstance(item, (list, tuple)) for item in a):
            return np.array(a)
        if all(isinstance(item, (int, float)) for item in a):
            return np.array([a])
        raise ValueError(f"Unsupported inputs: {a}")
    raise ValueError(f"Unsupported inputs: {a}")


def apply_color(target_list: list, c: list[str]):
    r"""make str colorful"""
    for i, target in enumerate(target_list):
        target = f'{target:.4f}' if isinstance(target, float) else target
        target_list[i] = f"\033[{c[i]}m{target}\033[0m"
    return target_list


def apply_format(target_list: list):
    r"""format bubble print"""
    s = f'{target_list[0]:^22}'
    symbol = ['=', '+', '+', '+', '+', '+']
    for i in range(len(target_list) - 1):
        s = f'{s}{symbol[i]}{target_list[i + 1]:^22}'
    return s


def color_mix(c1, c2, w1=0.5, w2=0.5):
    r"""mix colors"""
    rgb = (np.array(colors.to_rgba(c1, 1)) * w1 + np.array(colors.to_rgba(c2, 1)) * w2) / (w1 + w2)
    return colors.to_rgba(rgb)


def dfs_builder(comm=False):
    r"""build dfs wrapper"""
    def decorator(func):
        # pylint: disable=R1710
        def wrapper(*args, **kwargs):
            self = args[0]
            pre, left = (self.depend_pre, self.depend_left) if comm else (self.pre, self.left)
            if self.finish:
                return
            if pre is None or left is None:
                raise NotImplementedError
            if self.in_queue:
                raise ValueError
            self.in_queue = True
            res = func(*args, **kwargs)
            self.finish = True
            self.in_queue = False
            return res
        return wrapper
    return decorator


def timer(func):
    r"""timing"""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time() - t0
        print(f"function `{func.__name__}` time used: {t1:.4f} s", flush=True)
        return res
    return wrapper
