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
"""test api stability"""

import importlib
import inspect
import os
import pkgutil
import json
import re
from pathlib import Path
from types import BuiltinFunctionType
from typing import Callable
from itertools import chain
import pytest
import mindspore as ms
import mindformers


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def is_not_compatibility(base_str, new_str):
    """check whether compatibility"""
    base_io_params = base_str.split("->")
    new_io_params = new_str.split("->")
    base_input_params = base_io_params[0].strip()
    new_input_params = new_io_params[0].strip()
    base_out_params = "" if len(base_io_params) == 1 else base_io_params[1].strip()
    new_out_params = "" if len(new_io_params) == 1 else new_io_params[1].strip()

    # output params
    if base_out_params != new_out_params:
        if re.search("tuple.*common.tensor.Tensor.*common.tensor.Tensor.", base_out_params):  # special case
            return False
        return True

    base_params = base_input_params[1:-1].split(",")
    new_params = new_input_params[1:-1].split(",")
    base_diff_params = set(base_params) - set(new_params)

    new_diff_params = set(new_params) - set(base_params)
    if base_diff_params or new_diff_params:
        return True

    base_arr = [elem for elem in base_params if "=" not in elem]
    new_arr = [elem for elem in new_params if "=" not in elem]
    i = 0
    while i < len(base_arr):
        if base_arr[i] != new_arr[i]:
            return True
        i += 1

    return False


def set_failure_list(api_str, value, signature, failure_list):
    """set failure info list"""
    failure_list.append(f"# {api_str}:")
    failure_list.append(f"  - function signature is different: ")
    failure_list.append(f"    - the base signature is {value}.")
    failure_list.append(f"    - now it is {signature}.")


def process_union_order(signature):
    """process Union order, Union[str, bool] -> Union[bool, str]"""
    signature = re.sub(r", *", ", ", signature)
    union_area = []
    union_start = [match.start() for match in re.finditer(r"Union", signature)]
    for index in union_start:
        cur_union_end = None
        left_bracket_num, right_bracket_num = 0, 0
        for i, s in enumerate(signature[index:]):
            if s == "[":
                left_bracket_num += 1
            elif s == "]":
                right_bracket_num += 1
                if left_bracket_num == right_bracket_num:
                    cur_union_end = i + index
                    break
            elif s == ",":
                if left_bracket_num == right_bracket_num + 1:
                    # '~' is used for replace ',' temporarily
                    signature = signature[:i+index] + "!" + signature[i+index+1:]
        union_area.append((index, cur_union_end))
        if cur_union_end is None:
            raise ValueError("Not enough ] in signature.")
    union_str_len = 6  # used to extract list
    for area in union_area[::-1]:
        item_in_union = signature[area[0] + union_str_len: area[1]]
        item_list = item_in_union.split("! ")
        item_list.sort()
        item = f"Union[{'~ '.join(item_list)}]"     # '~' is used for replace ',' temporarily
        signature = signature[:area[0]] + item + signature[area[1]+1:]
    signature = re.sub(r"~", ",", signature)
    return signature


def api_signature(obj, api_str, content, base_schema, failure_list, is_update=False):
    """extract and compare api input info"""
    signature = inspect.signature(obj)
    signature = str(signature)
    if re.search("Any = <module 'pickle' from .+.py'>", signature):
        signature = re.sub("Any = <module 'pickle' from .+\\.py'>", "Any = <module 'pickle'>", signature)
    if re.search(" at 0x[\\da-zA-Z]+>", signature):
        signature = re.sub(" at 0x[\\da-zA-Z]+>", ">", signature)
    if re.search("<module.*dtype.py.>", signature):
        signature = re.sub("<module.*dtype.py.>", "mindspore.common.dtype", signature)
    if re.search(r"Union\[.*\]", signature):
        signature = process_union_order(signature)
    if is_update:
        content[api_str] = {"signature": signature}
    else:
        if api_str in base_schema.keys():
            value = base_schema[api_str]["signature"]
            if is_not_compatibility(value, signature):
                set_failure_list(api_str, value, signature, failure_list)


def is_mod_public(modname):
    """check whether module is public"""
    split_strs = modname.split('.')
    for elem in split_strs:
        if elem.startswith("_"):
            return False
    return True


def discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages.
    """
    for dir_path, _, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (pkg_path for _, pkg_path, _ in pkgutil.walk_packages((str(pkg_dir_path),), prefix=f'{pkg_pref}.',))


def find_all_importables(pkg):
    """Find all importables in the project.

    Return them in order.
    """
    return sorted(
        set(
            chain.from_iterable(
                discover_path_importables(Path(p), pkg.__name__)
                for p in pkg.__path__
            ),
        ),
    )


def func_in_class(obj, content, modname, elem, base_schema, failure_list, is_update=False):
    """check function in class"""
    class_variables = []
    for attribute in obj.__dict__.keys():
        if not attribute.startswith('__') and callable(getattr(obj, attribute)):
            class_variables.append(attribute)
    for variable in class_variables:
        if variable in ['_generate_next_value_', '_member_type_']:
            continue
        func = getattr(obj, variable)
        api_str = f"{modname}.{elem}.{variable}"
        api_signature(func, api_str, content, base_schema, failure_list, is_update)


class TestApiStability:
    """ A class which test api stability"""

    def setup_method(self):
        """init the class"""
        self.api_json_path = os.path.join(CUR_DIR_PATH, "base_schema.json")
        self.is_update = False  # when base_schema.json needs to update, set this value to True
        if not self.is_update:
            with open(self.api_json_path, "r", encoding="utf-8") as r:
                self.base_schema = json.loads(r.read())
        else:
            self.base_schema = {}

        self.all_importables = find_all_importables(mindformers)
        self.content = {}
        self.failure_list = []

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_modules(self):
        """test modules"""
        for mod_name in self.all_importables:
            if "__main__" in mod_name:
                continue
            try:
                mod = importlib.import_module(mod_name)
            except ModuleNotFoundError:
                # It is ok to ignore here as some modules' import packages are not in requirements and public
                # apis are not deply on these package
                continue

            if not is_mod_public(mod_name):
                continue

            def check_one_element(elem, mod_name, mod, is_public):
                obj = getattr(mod, elem)
                if hasattr(obj, "__module__"):
                    if obj.__module__ not in ['sentencepiece_model_pb2']:   # cannot use __import__ module list
                        mod_source = str(__import__(obj.__module__))
                        if "mindformers" not in mod_source:
                            return
                if not (isinstance(obj, (Callable, ms.dtype.TensorType)) or inspect.isclass(obj)) \
                        or isinstance(obj, BuiltinFunctionType):
                    return
                elem_module = getattr(obj, '__module__', None)

                elem_modname_starts_with_mod = \
                    elem_module is not None and elem_module.startswith(mod_name) and '._' not in elem_module

                is_public_api = not elem.startswith('_') and elem_modname_starts_with_mod
                if is_public != is_public_api:
                    is_public_api = is_public
                api_str = f"{mod_name}.{elem}"
                if api_str not in self.base_schema and not self.is_update:
                    return
                if is_public_api:
                    api_signature(obj, api_str, self.content, self.base_schema, self.failure_list, self.is_update)

                    # function in class
                    if inspect.isclass(obj):
                        func_in_class(obj, self.content, mod_name, elem, self.base_schema, self.failure_list,
                                      self.is_update)

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, mod_name, mod, is_public=elem in public_api)

        if not self.is_update:
            msg = "All the APIs below do not meet the compatibility guidelines. "
            msg += "If the change timeline has been reached, you can modify the base_schema.json to make it OK."
            msg += "\n\nFull list:\n"
            msg += "\n".join(map(str, self.failure_list))
            # empty lists are considered false in python
            if self.failure_list:
                raise AssertionError(msg)
        else:
            with open(self.api_json_path, "w", encoding="utf-8") as w:
                w.write(json.dumps(self.content, ensure_ascii=False, indent=4))
