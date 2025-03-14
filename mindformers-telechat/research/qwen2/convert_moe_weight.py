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
Convert qwen2 moe weight.
Support huggingface format
"""

import argparse
from pathlib import Path
import re

import mindspore as ms
import numpy as np
import torch
from safetensors.torch import load_file
from mindformers.tools.utils import str2bool
from mindformers.utils.convert_utils import pt2ms, qkv_concat_hf2mg
from mindformers.tools import MindFormerConfig, MindFormerRegister, MindFormerModuleType

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def get_expert_id(name):
    res = re.search(r'experts.ffn.(\d+).', name)
    if res:
        return int(res.group(1))
    return -1


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.mlp.gate.', '.feed_forward.routed_experts.router.dense.')
    name = name.replace('.mlp.experts.', '.feed_forward.routed_experts.ffn.')
    name = name.replace('.mlp.shared_expert.gate_proj.weight', '.feed_forward.shared_experts.w1.weight')
    name = name.replace('.mlp.shared_expert.up_proj.weight', '.feed_forward.shared_experts.w3.weight')
    name = name.replace('.mlp.shared_expert.down_proj.weight', '.feed_forward.shared_experts.w2.weight')
    name = name.replace('.mlp.shared_expert_gate.weight', '.feed_forward.shared_experts_gate.weight')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('model.norm.weight', 'model.norm_out.weight')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, use_gmm=False, config_path=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.")
    config = MindFormerConfig(config_path)
    if 'auto_register' in config.model:
        MindFormerRegister.auto_register(class_reference=config.model.pop('auto_register'),
                                         module_type=MindFormerModuleType.MODELS)
    try:
        ckpt_paths = sorted(Path(input_path).glob("*.safetensors"))
        dict_all = {}
        for i, _ in enumerate(ckpt_paths):
            state_dict = load_file(ckpt_paths[i], device='cpu')
            dict_all.update(state_dict)
        model_hf = dict(sorted(dict_all.items(), key=lambda x: x[0]))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{input_path}', Error {e.message}.")
        return False
    params = dict()
    count = 0
    list_w1, list_w2, list_w3 = [], [], []
    expert_num = config.moe_config.expert_num
    for name, value in model_hf.items():
        name = name_replace(name)
        if 'feed_forward.routed_experts.ffn' not in name:
            print(f'\rprocessing parameter: {name} {value.shape}     ')
            params[name] = ms.Parameter(pt2ms(value, dtype))
        else:
            if count % 3 == 0 and count != expert_num * 3:
                list_w2.append({"name": name, "value": value})
            if count % 3 == 1 and count != expert_num * 3:
                list_w1.append({"name": name, "value": value})
            if count % 3 == 2 and count != expert_num * 3:
                list_w3.append({"name": name, "value": value})
            count = count + 1
            if count == expert_num * 3:
                # list_w1, list_w2, list_w3排序
                list_w1.sort(key=lambda x: get_expert_id(x["name"]))
                list_w1 = [item["value"] for item in list_w1]
                list_w2.sort(key=lambda x: get_expert_id(x["name"]))
                list_w2 = [item["value"] for item in list_w2]
                list_w3.sort(key=lambda x: get_expert_id(x["name"]))
                list_w3 = [item["value"] for item in list_w3]
                str_front = name.split('ffn')[0]
                stack_routed_experts_weight(params=params, list_w=list_w1, name_w=str_front + 'ffn.w1.weight',
                                            use_gmm=use_gmm, dtype=dtype)
                stack_routed_experts_weight(params=params, list_w=list_w2, name_w=str_front + 'ffn.w2.weight',
                                            use_gmm=use_gmm, dtype=dtype)
                stack_routed_experts_weight(params=params, list_w=list_w3, name_w=str_front + 'ffn.w3.weight',
                                            use_gmm=use_gmm, dtype=dtype)
                count = 0
                list_w1, list_w2, list_w3 = [], [], []
    qkv_concat = config.model.model_config.get("qkv_concat", True)
    if qkv_concat:
        concat_weight_and_bias(params, config.model)
    ms.save_checkpoint(params, output_path)
    print(f"\rConvert finished, the mindspore ckpt is saved in '{output_path}'.")
    return True


def convert_ms_to_gmm(input_path, output_path, **kwargs):
    """convert ms weight to gmm."""
    params = ms.load_checkpoint(input_path)
    for k, v in params.items():
        if 'feed_forward.ffn.w1.weight' in k or \
                'feed_forward.ffn.w2.weight' in k or \
                'feed_forward.ffn.w3.weight' in k:
            orig_tensor = ms.Tensor(v)
            gmm_tensor = orig_tensor.transpose((0, 2, 1))
            params[k] = ms.Parameter(gmm_tensor)
            print(f"\rConvert {params[k]} to gmm weight.", flush=True)
    ms.save_checkpoint(params, output_path)
    print(f"\rConvert finished, the mindspore ckpt is saved in '{output_path}'.")
    return True


def concat_param(param_dict, param_name_list, concat_name, config):
    """convert qkv weights to megatron format"""
    num_heads = config.model_config.get('num_heads')
    n_kv_heads = config.model_config.get('n_kv_heads', None) or num_heads
    hidden_size = config.model_config.get('hidden_size')
    param_value_list = list()
    for param_name in param_name_list:
        param_value_list.append(param_dict[param_name].asnumpy())
        param_dict.pop(param_name)
    concat_value = np.concatenate(param_value_list, 0)
    concat_value = qkv_concat_hf2mg(concat_value, num_heads, n_kv_heads, hidden_size)
    param_dict[concat_name] = ms.Parameter(concat_value, name=concat_name)
    print("transform: {}".format(concat_name))


def stack_routed_experts_weight(params, list_w, name_w, dtype, use_gmm):
    value_w = torch.stack(list_w, 0)
    if use_gmm:
        value_w = torch.permute(value_w, (0, 2, 1))
    print(f'\rprocessing parameter: {name_w} {value_w.shape}     ')
    params[name_w] = ms.Parameter(pt2ms(value_w, dtype))


def concat_weight_and_bias(param_dict, config):
    """concat qkv weight"""
    qkv_weight_name = "model.layers.{}.attention.{}.weight"
    qkv_bias_name = "model.layers.{}.attention.{}.bias"
    for i in range(config.model_config.num_layers):
        # qkv weight concat
        qkv_weight_param_name_list = [qkv_weight_name.format(i, "wq"),
                                      qkv_weight_name.format(i, "wk"),
                                      qkv_weight_name.format(i, "wv")]
        qkv_weight_concat_name = qkv_weight_name.format(i, "w_qkv")
        concat_param(param_dict=param_dict,
                     param_name_list=qkv_weight_param_name_list,
                     concat_name=qkv_weight_concat_name,
                     config=config)
        qkv_bias_param_name_list = [qkv_bias_name.format(i, "wq"),
                                    qkv_bias_name.format(i, "wk"),
                                    qkv_bias_name.format(i, "wv")]
        qkv_bias_concat_name = qkv_bias_name.format(i, "w_qkv")
        concat_param(param_dict=param_dict,
                     param_name_list=qkv_bias_param_name_list,
                     concat_name=qkv_bias_concat_name,
                     config=config)
    return param_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./qwen2/torch_ckpt/')
    parser.add_argument('--mindspore_ckpt_path', default='./qwen2/ms_ckpt/')
    parser.add_argument('--use_gmm', type=str2bool, default=True)
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--config_path', default=None)
    args = parser.parse_args()

    if args.pre_ckpt_path:
        convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
    else:
        dtype_src = dtype_map.get(args.dtype)
        if dtype_src:
            convert_pt_to_ms(input_path=args.torch_ckpt_dir, output_path=args.mindspore_ckpt_path,
                             dtype=dtype_src, use_gmm=args.use_gmm, config_path=args.config_path)
        else:
            raise ValueError(f"args.dtype:{args.dtype} is not in dtype_map:{dtype_map}.")
