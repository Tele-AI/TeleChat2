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
"""safetensors utils"""
import os

from safetensors import safe_open

from mindformers.tools import logger


def is_hf_safetensors_dir(safetensors_dir, model_cls_or_instance):
    """Is HuggingFace safetensors directory"""
    if not contains_safetensors_files(safetensors_dir):
        return False

    sf_list = [sf for sf in os.listdir(safetensors_dir) if sf.endswith('.safetensors')]
    for sf in sf_list:
        with safe_open(os.path.join(safetensors_dir, sf), framework="np") as f:
            all_keys = f.keys()
        for key in all_keys:
            # once convert success
            try:
                if key != model_cls_or_instance.convert_name(key):
                    return True
            except RuntimeError:
                print("The model does not have a convert_name method, "
                      "please make sure your safetensors are converted to ms type.")
    return False


def contains_safetensors_files(safetensors_dir):
    """whether the given directory contains safetensors files"""
    if not os.path.isdir(safetensors_dir):
        return False

    for filename in os.listdir(safetensors_dir):
        if filename.endswith('.safetensors'):
            return True

    return False


def check_safetensors_key(load_checkpoint_dir, key):
    """Check if there are any key names containing the character "key" in safetensors files"""
    sf_list = [sf for sf in os.listdir(load_checkpoint_dir) if sf.endswith('.safetensors')]
    for sf in sf_list:
        with safe_open(os.path.join(load_checkpoint_dir, sf), framework="np") as f:
            all_items = f.keys()
        has_key = any(key in item for item in all_items)
        if has_key:
            logger.debug("safetensors %s containing %s", sf, key)
            return True

    return False
