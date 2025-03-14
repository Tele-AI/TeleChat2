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
"""convert safetensors"""
import json
import os
import shutil
from multiprocessing import Process, Manager, Condition
from safetensors.numpy import load_file, save_file

from mindformers.tools import logger
from .utils import is_hf_safetensors_dir


def convert_hf_safetensors_multiprocess(src_dir, dst_dir, model_cls_or_instance, model_config):
    """Convert HuggingFace safetensors to MindSpore safetensors with multiprocessing"""
    _check_valid_input(src_dir, dst_dir, model_cls_or_instance, model_config)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    logger.info("Folder %s is remade.", dst_dir)
    logger.info(".........Starting to Convert Safetensors.........")
    # convert safetensors
    _convert_safetensors(src_dir,
                         dst_dir,
                         model_cls_or_instance.convert_weight_dict,
                         model_config)
    # convert json
    _convert_index_json(src_dir, dst_dir, model_cls_or_instance.convert_map_dict, model_config.qkv_concat)
    logger.info(".........Safetensors Convert Complete.........")


def _check_valid_input(src_dir, dst_dir, model_cls_or_instance, model_config):
    """check whether the input arguments are valid"""
    num_heads = model_config.num_heads
    n_kv_heads = model_config.n_kv_heads
    hidden_size = model_config.hidden_size
    qkv_concat = model_config.qkv_concat

    if not isinstance(src_dir, str) or isinstance(src_dir, os.PathLike):
        raise ValueError(f"src_dir must be a str or an instance of os.PathLike, "
                         f"but got {src_dir} as type {type(src_dir)}.")
    if not isinstance(dst_dir, str):
        raise ValueError(f"src_dir must be a str or an instance of os.PathLike, "
                         f"but got {dst_dir} as type {type(dst_dir)}.")
    from mindformers.models.modeling_utils import PreTrainedModel
    if not (isinstance(model_cls_or_instance, PreTrainedModel) or
            isinstance(model_cls_or_instance, type) and issubclass(model_cls_or_instance, PreTrainedModel)):
        raise ValueError(f"model_cls_or_instance must be a subclass or an instance of PreTrainedModel,"
                         f"but got {model_cls_or_instance}.")
    if not is_hf_safetensors_dir(src_dir, model_cls_or_instance):
        raise ValueError(f"src_dir is not a valid HuggingFace safetensors directory.")
    if not isinstance(num_heads, int):
        raise ValueError(f"num_attention_heads must be an int value, but got {num_heads}.")
    if n_kv_heads and not isinstance(n_kv_heads, int):
        raise ValueError(f"kv_heads must be an int value, but got {n_kv_heads}.")
    if not isinstance(hidden_size, int):
        raise ValueError(f"hidden must be an int value, but got {hidden_size}.")
    if not isinstance(qkv_concat, bool):
        raise ValueError(f"is_qkv_concat must be a bool value, but got {qkv_concat}.")


def _convert_safetensors(load_checkpoint, converted_dir, convert_weight_dict, model_config):
    """Create multiprocess to convert the safetensors"""
    sf_list = [sf for sf in os.listdir(load_checkpoint) if sf.endswith('.safetensors')]
    processes = []
    qkv_dict = None
    condition = None
    if model_config.qkv_concat:
        manager = Manager()
        qkv_dict = manager.dict()
        condition = Condition()
    for sf in sf_list:
        p = Process(target=_convert_process, args=[os.path.join(load_checkpoint, sf),
                                                   os.path.join(converted_dir, sf),
                                                   convert_weight_dict,
                                                   model_config,
                                                   qkv_dict,
                                                   condition])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def _convert_index_json(load_checkpoint, converted_dir, convert_map_dict, is_qkv_concat):
    """convert mapping file if exists"""
    index_path = os.path.join(load_checkpoint, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        logger.warning(f"The given path contains no 'model.safetensors.index.json' file.")
        return
    with open(index_path, 'r') as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    new_weight_map = convert_map_dict(weight_map, qkv_concat=is_qkv_concat)
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(os.path.join(converted_dir, 'param_name_map.json'), flags_, 0o750), 'w') as f:
        json.dump(new_weight_map, f, indent=2)
        logger.info(f"Converted file param_name_map.json")


def _convert_process(src_dir, dst_dir, convert_weight_dict, model_config, qkv_dict=None, condition=None):
    """A single process to convert the safetensors"""
    source_dict = load_file(src_dir)
    target_dict = convert_weight_dict(source_dict, model_config=model_config, qkv_dict=qkv_dict, condition=condition)
    save_file(tensor_dict=target_dict, filename=dst_dir)
    logger.info(f"Converted file {os.path.basename(dst_dir)}.")
