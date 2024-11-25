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
Convert llama weight.
Support huggingface format and Meta format.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import ops

from mindformers.tools.utils import str2bool
from mindformers.utils.convert_utils import pt2ms


def convert_meta_torch_ckpt(ckpt_dir, output_name, dtype=ms.float16):
    """Support convert meta weight splited."""
    print(f"Trying to convert pytorch checkpoint in '{ckpt_dir}'.", flush=True)
    try:
        from torch import load
    except ModuleNotFoundError as e:
        raise ImportError("Failed to load PyTorch checkpoint. "
                          "PyTorch module is not installed. Please make sure PyTorch is available.") from e
    except ImportError as e:
        raise ImportError("Failed to load PyTorch checkpoint. "
                          "There was an error while importing the PyTorch module.") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred when loading PyTorch checkpoint. Error: {str(e)}") from e
    dic = {
        'tok_embeddings.weight': 1,
        'norm.weight': None,
        'output.weight': 0,
        'attention.wq.weight': 0,
        'attention.wk.weight': 0,
        'attention.wv.weight': 0,
        'attention.wo.weight': 1,
        'feed_forward.w1.weight': 0,
        'feed_forward.w2.weight': 1,
        'feed_forward.w3.weight': 0,
        'attention_norm.weight': None,
        'ffn_norm.weight': None,
        'rope.freqs': None,
    }
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if not ckpt_paths:
        print(f"Do not find pytorch checkpoint in '{ckpt_dir}'.", flush=True)
        return False
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        model_args = json.loads(f.read())
    n_heads = model_args["n_heads"]
    dim = model_args["dim"]

    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    checkpoints = []
    for i, _ in enumerate(ckpt_paths):
        checkpoints.append(load(ckpt_paths[i], map_location="cpu"))
    ckpt_list = []
    for name in checkpoints[0].keys():
        for k, v in dic.items():
            if k in name:
                if v is not None:
                    value = np.concatenate(
                        [checkpoints[i][name].numpy() for i in range(len(checkpoints))], v)
                else:
                    value = checkpoints[0][name].numpy()
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        else:
            name = 'model.' + name
        if 'rope.freqs' in name:
            continue

        if 'wq' in name or 'wk' in name:
            value = permute(value)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ckpt_file = os.path.join(ckpt_dir, output_name)
    ms.save_checkpoint(ckpt_list, ckpt_file)
    print(f"\rConvert pytorch checkpoint finished, the mindspore checkpoint is saved in '{ckpt_file}'.", flush=True)
    return True


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')
    return name


def flatten_dict(ckpt, parent_key='', sep='.'):
    """flatten the ckpt dict."""
    items = []
    for k, v in ckpt.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# pylint: disable=W0613
def convert_megatron_to_ms(input_path, output_path, dtype=None, **kwargs):
    """ Convert megatron ckpt to mindspore ckpt """
    print(f"Trying to convert megatron checkpoint in '{input_path}'.", flush=True)
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Failed to load PyTorch checkpoint. Please make sure PyTorch is installed and available."
        ) from e
    except Exception as e:
        raise RuntimeError("Unexpected error occurred when loading PyTorch checkpoint.") from e
    try:
        megatron_ckpt = torch.load(input_path, map_location='cpu')
    # pylint: disable=W0703
    except Exception as e:
        print(f"Fail to load meagtron checkpoint '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False

    megatron_keys = flatten_dict(megatron_ckpt.get('model'))
    key_mapping = {
        'dense_h_to_4h': 'mapping',
        'dense_4h_to_h': 'projection',
        'self_attention': 'attention',
        'query_key_value': 'qkv_proj',
        'dense': 'out_proj',
    }
    ms_ckpt = {}
    for key, value in megatron_keys.items():
        if '_extra_state' in key:
            continue
        ms_key = key
        for k, v in key_mapping.items():
            if k in ms_key:
                ms_key = ms_key.replace(k, v)
        ms_ckpt[ms_key] = value

    ckpt_list = []
    for k, v in ms_ckpt.items():
        print(f'\rprocessing parameter: {k} {v.shape}     ', end='', flush=True)
        ckpt_list.append({'name': k, 'data': pt2ms(v, dtype)})
    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert megatron checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import LlamaForCausalLM
    except ImportError as e:
        raise ImportError(
            "Failed to load Hugging Face checkpoint. "
            "Please make sure the 'transformers' library is installed and available."
        ) from e
    except Exception as e:
        raise RuntimeError("Unexpected error occurred when loading Hugging Face checkpoint.") from e
    try:
        model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(input_path))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


def convert_to_new_ckpt(ckpt_path, config_path):
    """convert previous ckpt to new ckpt"""
    load_path = ckpt_path.split('.ckpt')[0]
    save_path = load_path + "_hf"
    params = ms.load_checkpoint(load_path.split('.ckpt')[0] + '.ckpt')
    with open(config_path, "r") as f:
        model_args = json.loads(f.read())
    n_heads = model_args["n_heads"]
    dim = model_args["dim"]

    def permute(w):
        return ops.transpose(w.reshape(n_heads, dim // n_heads // 2, 2, dim), (0, 2, 1, 3)).reshape(dim, dim)

    ckpt_list = []
    for name in params.keys():
        value = params[name].value()
        if '.wq' in name or '.wk' in name:
            value = permute(value)
        ckpt_list.append({'name': name, 'data': value})
        print("\r", name, value.shape, end="               ")

    ms.save_checkpoint(ckpt_list, save_path)


def convert_qkv_concat_weight(param_dict):
    """convert qkv concat weight"""
    assume_num_layers = 500
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
        qkv_weight = np.concatenate((wq_weight, wk_weight, wv_weight), 0)
        param_dict[qkv_concat_weight_name] = ms.Parameter(qkv_weight, name=qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"

        ffn_gate_weight = param_dict[ffn_gate_weight_name].asnumpy()
        ffn_hidden_weight = param_dict[ffn_hidden_weight_name].asnumpy()
        gate_hidden_weight = np.concatenate((ffn_gate_weight, ffn_hidden_weight), 0)
        param_dict[gate_hidden_concat_weight_name] = ms.Parameter(gate_hidden_weight,
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
        qkv_bias_weight = np.concatenate((wq_bias_weight, wk_bias_weight, wv_bias_weight), 0)
        param_dict[qkv_concat_bias_name] = ms.Parameter(qkv_bias_weight, name=qkv_concat_bias_name)

        param_dict.pop(wq_bias_name)
        param_dict.pop(wk_bias_name)
        param_dict.pop(wv_bias_name)
        print("transform: {}".format(qkv_concat_bias_name))
    return param_dict


def convert_to_qkv_concat(pre_ckpt_path, mindspore_ckpt_path):
    """convert previous ckpt to qkv concat ckpt"""
    if os.path.isdir(pre_ckpt_path):
        rank_dir_list = os.listdir(pre_ckpt_path)
        for rank_dir in rank_dir_list:
            rank_dir_name = str(rank_dir)
            rank_id = rank_dir_name.split("_")[1]
            checkpoint_path = os.path.join(pre_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            print("checkpoint_path: {}".format(checkpoint_path))
            params = ms.load_checkpoint(checkpoint_path)
            params = convert_qkv_concat_weight(params)

            save_dir = os.path.join(mindspore_ckpt_path, rank_dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(mindspore_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            ms.save_checkpoint(params, save_path)
    else:
        params = ms.load_checkpoint(pre_ckpt_path)
        params = convert_qkv_concat_weight(params)
        ms.save_checkpoint(params, mindspore_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_path', default='./llama_model/llama-13b-hf/hf.bin')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--config_path', default=None)
    parser.add_argument('--qkv_concat', default=False, type=str2bool)
    args = parser.parse_args()

    if args.qkv_concat:
        convert_to_qkv_concat(args.pre_ckpt_path, mindspore_ckpt_path=args.mindspore_ckpt_path)
    elif args.pre_ckpt_path is not None and args.config_path is not None:
        convert_to_new_ckpt(args.pre_ckpt_path, args.config_path)
    else:
        convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path)
