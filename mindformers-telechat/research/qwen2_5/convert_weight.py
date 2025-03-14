# Copyright 2025 Huawei Technologies Co., Ltd
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
transform huggingface model to mindspore ckpt.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import mindspore as ms

from mindformers import MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import str2bool
from mindformers.utils.convert_utils import qkv_concat_hf2mg, ffn_concat_hf2mg

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('lm_head.', 'output.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
    weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
    weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace(
        '.post_attention_layernorm.', '.ffn_norm.')
    # Required for lora weight conversion
    weight_name = weight_name.replace('base_model.model.', '')
    weight_name = weight_name.replace('lora_A.weight', 'lora_a')
    weight_name = weight_name.replace('lora_B.weight', 'lora_b')
    return weight_name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(
        f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError as e:
        raise ImportError(
            "Failed to load HuggingFace checkpoint. "
            "Please make sure the 'transformers' library is installed and available."
        ) from e

    try:
        model_hf = Qwen2ForCausalLM.from_pretrained(
            os.path.dirname(input_path))
    # pylint: disable=W0703
    except Exception as e:
        print(
            f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'
        value = value.detach().numpy()
        print(name, value.shape)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


def convert_lora_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(
        f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "Failed to load HuggingFace checkpoint. "
            "Please make sure the 'safetensors' library is installed and available."
        ) from e

    try:
        ckpt_paths = sorted(Path(input_path).glob("adapter_model.safetensors"))
        dict_all = {}
        for ckpt_path in ckpt_paths:
            state_dict = load_file(ckpt_path, device='cpu')
            dict_all.update(state_dict)
        model_hf = dict(sorted(dict_all.items(), key=lambda x: x[0]))
    # pylint: disable=W0703
    except Exception as e:
        print(
            f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.items():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'
        value = value.detach().numpy()
        print(name, value.shape)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    convert_lora_config(input_path, kwargs['align_rank'])
    return True


def convert_lora_config(input_path, align_rank):
    """modified config.json 'r' and 'target_modules' """
    config_path = os.path.join(input_path, "adapter_config.json")
    replace_rules = {
        'q_proj': 'wq',
        'k_proj': 'wk',
        'v_proj': 'wv',
        'o_proj': 'wo',
        'gate_proj': 'w1',
        'down_proj': 'w2',
        'up_proj': 'w3'
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        modified = False
        # r must be a multiple of 16 on Atlas 300V Pro
        if align_rank:
            if data["r"] % 16 != 0:
                data["r"] = (data["r"] // 16 + 1) * 16
                print("r modification successful, the configuration file has been updated!")
            else:
                print("The configuration r has already been modified, no need to modify it again.")

        for i, name in enumerate(data["target_modules"]):
            if name not in replace_rules.keys():
                print(f"target_modules {name} does not need to be modified")
            else:
                data["target_modules"][i] = replace_rules[name]
                print(f"target_modules {name} has been modified to {replace_rules[name]}")
                modified = True

        if modified:
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            print("Target_modules modification successful, the configuration file has been updated!")
        else:
            print("The configuration target_modules has already been modified, no need to modify it again.")

        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"JSON file modified successfully!")

    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
    except KeyError:
        print("Error: The specified key does not exist in the JSON")
    except json.JSONDecodeError:
        print("Error: File content is not valid JSON format")


def convert_qkv_concat_weight(param_dict, model_config):
    """convert qkv concat weight"""
    assume_num_layers = 500
    num_heads = model_config.model_config.get('num_heads')
    n_kv_heads = model_config.model_config.get('n_kv_heads', None) or num_heads
    hidden_size = model_config.model_config.get('hidden_size')
    ffn_hidden_size = 4 * hidden_size
    intermediate_size = model_config.model_config.get('intermediate_size', None)
    ffn_dim_multiplier = model_config.model_config.get('ffn_dim_multiplier', None)
    multiple_of = model_config.model_config.get('multiple_of', 256)
    if intermediate_size is not None:
        ffn_hidden_size = intermediate_size
    else:
        if ffn_dim_multiplier is not None:
            ffn_hidden_size = int((ffn_dim_multiplier + 0.01) * ffn_hidden_size)
        ffn_hidden_size = int(2 * ffn_hidden_size / 3)
        ffn_hidden_size = multiple_of * \
                          ((ffn_hidden_size + multiple_of - 1) // multiple_of)
    for i in range(assume_num_layers):
        # qkv weight concat
        wq_weight_name = f"model.layers.{i}.attention.wq.weight"
        wk_weight_name = f"model.layers.{i}.attention.wk.weight"
        wv_weight_name = f"model.layers.{i}.attention.wv.weight"
        qkv_concat_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
        if wq_weight_name not in param_dict:
            break
        wq_weight = param_dict[wq_weight_name].asnumpy()
        wk_weight = param_dict[wk_weight_name].asnumpy()
        wv_weight = param_dict[wv_weight_name].asnumpy()
        qkv_weight_hf = np.concatenate((wq_weight, wk_weight, wv_weight), 0)
        # qkv weight format: hf -> mg
        qkv_weight_mg = qkv_concat_hf2mg(qkv_weight_hf, num_heads, n_kv_heads, hidden_size)
        param_dict[qkv_concat_weight_name] = ms.Parameter(qkv_weight_mg, name=qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"

        ffn_gate_weight = param_dict[ffn_gate_weight_name].asnumpy()
        ffn_hidden_weight = param_dict[ffn_hidden_weight_name].asnumpy()
        gate_hidden_weight_hf = np.concatenate((ffn_gate_weight, ffn_hidden_weight), 0)
        # ffn weight format: hf -> mg
        gate_hidden_weight_mg = ffn_concat_hf2mg(gate_hidden_weight_hf, ffn_hidden_size)
        param_dict[gate_hidden_concat_weight_name] = ms.Parameter(gate_hidden_weight_mg,
                                                                  name=gate_hidden_concat_weight_name)

        param_dict.pop(wq_weight_name)
        param_dict.pop(wk_weight_name)
        param_dict.pop(wv_weight_name)
        param_dict.pop(ffn_gate_weight_name)
        param_dict.pop(ffn_hidden_weight_name)
        print("transform: {}".format(qkv_concat_weight_name))
        print("transform: {}".format(gate_hidden_concat_weight_name))

    for i in range(assume_num_layers):
        # qkv bias concat
        wq_bias_name = f"model.layers.{i}.attention.wq.bias"
        wk_bias_name = f"model.layers.{i}.attention.wk.bias"
        wv_bias_name = f"model.layers.{i}.attention.wv.bias"
        qkv_concat_bias_name = f"model.layers.{i}.attention.w_qkv.bias"
        if wq_bias_name not in param_dict:
            break

        wq_bias_weight = param_dict[wq_bias_name].asnumpy()
        wk_bias_weight = param_dict[wk_bias_name].asnumpy()
        wv_bias_weight = param_dict[wv_bias_name].asnumpy()
        qkv_bias_weight_hf = np.concatenate((wq_bias_weight, wk_bias_weight, wv_bias_weight), 0)
        # qkv bias weight format: hf -> mg
        qkv_bias_weight_mg = qkv_concat_hf2mg(qkv_bias_weight_hf, num_heads, n_kv_heads, hidden_size)
        param_dict[qkv_concat_bias_name] = ms.Parameter(qkv_bias_weight_mg, name=qkv_concat_bias_name)

        param_dict.pop(wq_bias_name)
        param_dict.pop(wk_bias_name)
        param_dict.pop(wv_bias_name)
        print("transform: {}".format(qkv_concat_bias_name))
    return param_dict


def convert_to_qkv_concat(pre_ckpt_path, mindspore_ckpt_path, config_path):
    """convert previous ckpt to qkv concat ckpt"""
    model_config = MindFormerConfig(config_path).model
    if 'auto_register' in model_config:
        MindFormerRegister.auto_register(class_reference=model_config.pop('auto_register'),
                                         module_type=MindFormerModuleType.MODELS)

    if os.path.isdir(pre_ckpt_path):
        rank_dir_list = os.listdir(pre_ckpt_path)
        for rank_dir in rank_dir_list:
            rank_dir_name = str(rank_dir)
            rank_id = rank_dir_name.split("_")[1]
            checkpoint_path = os.path.join(pre_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            print("checkpoint_path: {}".format(checkpoint_path))
            params = ms.load_checkpoint(checkpoint_path)
            params = convert_qkv_concat_weight(params, model_config)

            save_dir = os.path.join(mindspore_ckpt_path, rank_dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(mindspore_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            ms.save_checkpoint(params, save_path)
    else:
        params = ms.load_checkpoint(pre_ckpt_path)
        params = convert_qkv_concat_weight(params, model_config)
        ms.save_checkpoint(params, mindspore_ckpt_path)


def convert_weight(para):
    """convert weight entrance"""
    if not hasattr(para, 'torch_ckpt_dir'):
        para.torch_ckpt_dir = para.input_path
    if not hasattr(para, 'pre_ckpt_path'):
        para.pre_ckpt_path = para.input_path
    if not hasattr(para, 'mindspore_ckpt_path'):
        para.mindspore_ckpt_path = para.output_path
    if not hasattr(para, 'config_path'):
        para.config_path = para.config_path
    if not para.dtype:
        para.dtype = "bf16"
    if para.qkv_concat:
        convert_to_qkv_concat(para.pre_ckpt_path, para.mindspore_ckpt_path, para.config_path)
    else:
        dtype = dtype_map.get(para.dtype)
        if para.is_lora:
            convert_lora_to_ms(input_path=para.torch_ckpt_dir, output_path=para.mindspore_ckpt_path,
                               dtype=dtype, align_rank=para.align_rank)
        else:
            convert_pt_to_ms(input_path=para.torch_ckpt_dir,
                             output_path=para.mindspore_ckpt_path, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--config_path', default=None)
    parser.add_argument('--qkv_concat', default=False, type=str2bool)
    parser.add_argument('--dtype', default='bf16')
    parser.add_argument('--is_lora', default=False)
    parser.add_argument('--align_rank', default=False)
    args = parser.parse_args()

    convert_weight(args)
