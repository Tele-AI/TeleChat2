# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024, All rights reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""
import os
import tempfile

from contextlib import contextmanager


@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir


def add_model_info_to_auto_map(auto_map, repo_id):
    """
    Adds the information of the repo_id to a given auto map.
    """
    for key, value in auto_map.items():
        if isinstance(value, (tuple, list)):
            auto_map[key] = [f"{repo_id}--{v}" if (v is not None and "--" not in v) else v for v in value]
        elif value is not None and "--" not in value:
            auto_map[key] = f"{repo_id}--{value}"

    return auto_map


def experimental_mode_func_checker(custom_err_msg=None):
    """Raise RuntimeError when detecting exception in decorated function.

    :param: func: decorated function
    :return: decorator
    """

    def wrapper(func):
        def inner_wrapper(cls, *args, **kwargs):
            try:
                return func(cls, *args, **kwargs)
            except Exception as e:
                error_msg = f"Error occurred when executing function {func.__name__}."

                if custom_err_msg:
                    error_msg += f"\n You are using {cls.__name__} in experimental mode, which is not available" \
                                 f" now. Check the error message: {custom_err_msg}"

                raise RuntimeError(error_msg) from e

        return inner_wrapper

    return wrapper


def is_json_file(path):
    return os.path.isfile(path) and path.endswith(".json")


def is_experimental_mode(path):
    """Check whether AutoConfig.from_pretrained() should go into original or experimental mode

    :param path: (str) path to AutoConfig.from_pretrained()
    :return: (bool) whether AutoConfig.from_pretrained() should go into original or experimental mode
    """
    experimental_mode = False

    if not isinstance(path, str):
        raise ValueError(f"param 'path' in AutoConfig.from_pretrained() must be str, but got {type(path)}")

    if not os.path.exists(path) and "/" in path and path.split("/")[0] != "mindspore":
        experimental_mode = True
    elif os.path.isdir(path) or is_json_file(path):
        experimental_mode = True

    return experimental_mode
