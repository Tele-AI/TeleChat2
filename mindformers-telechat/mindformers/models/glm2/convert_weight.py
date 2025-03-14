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
"""Convert checkpoint from torch/huggingface"""

import os
import argparse
from typing import List

import mindspore as ms
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from mindformers.tools import MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import str2bool
from mindformers.models.glm2.glm2_config import ChatGLM2Config
from mindformers.utils.convert_utils import pt2ms


def npy2ms(arr: np.array, dtype):
    """npy2ms"""
    return ms.Tensor(arr, dtype=dtype)


def rearrange_w(w: np.ndarray, head_dim: int):
    """rearrange w"""
    w = w.T
    hidden_size, projection_size = w.shape
    w = w.reshape(hidden_size, projection_size // head_dim, head_dim)
    w = np.concatenate(
        [
            w[..., 0::2],
            w[..., 1::2]
        ],
        axis=-1
    )
    w = w.reshape(hidden_size, projection_size)
    return w.T


def rearrange_b(b: np.ndarray, head_dim: int):
    """rearrange b"""
    projection_size = b.shape[0]
    b = b.reshape(projection_size // head_dim, head_dim)
    b = np.concatenate(
        [
            b[..., 0::2],
            b[..., 1::2],
        ],
        axis=-1
    )
    b = b.reshape(-1,)
    return b


def attn_split(param_list: List, config: ChatGLM2Config, dtype):
    """attention split"""
    is_mqa = config.multi_query_attention  # Multi query attention
    kv_channels = config.kv_channels
    num_attention_heads = config.num_attention_heads
    projection_size = kv_channels * num_attention_heads
    n_kv_head = config.multi_query_group_num
    head_dim = kv_channels
    q_hidden_size = projection_size
    kv_hidden_size = head_dim * n_kv_head if is_mqa else projection_size

    param_idx_to_del = []
    param_split = []

    for idx, item in enumerate(param_list):
        name: str = item['name']
        data: torch.Tensor = item['data']
        if "query_key_value" not in name:
            continue
        param_idx_to_del.append(idx)
        org_dtype = data.dtype
        data = data.to(torch.float32).numpy() if org_dtype == torch.bfloat16 else data.numpy()
        # Split on axis 0 for enabling transpose_b
        wq, wk, wv = np.split(data, [q_hidden_size, q_hidden_size + kv_hidden_size], axis=0)
        w_name = name.replace("query_key_value", "wq")
        k_name = name.replace("query_key_value", "wk")
        v_name = name.replace("query_key_value", "wv")
        if config.use_rearrange_rope:
            # rearrange wq
            if len(wq.shape) == 2:
                wq = rearrange_w(wq, head_dim)
                wk = rearrange_w(wk, head_dim)
            else:
                wq = rearrange_b(wq, head_dim)
                wk = rearrange_b(wk, head_dim)
        if not config.qkv_concat:
            param_split.append({"name": w_name, "data": npy2ms(wq, dtype)})
            param_split.append({"name": k_name, "data": npy2ms(wk, dtype)})
            param_split.append({"name": v_name, "data": npy2ms(wv, dtype)})
        else:
            concat_qkv = np.concatenate(
                [wq, wk, wv],
                axis=0
            )
            param_split.append({"name": name, "data": npy2ms(concat_qkv, dtype)})
    for idx in reversed(param_idx_to_del):  # delete element in list by index
        param_list.pop(idx)

    param_list += param_split


def mlp_split(param_list: List, config: ChatGLM2Config, dtype):
    """mlp split"""
    ffn_hidden_size = config.ffn_hidden_size
    param_idx_to_del = []
    param_split = []

    for idx, item in enumerate(param_list):
        name: str = item['name']
        data: ms.Tensor = item['data']
        if "dense_h_to_4h" not in name:
            continue
        param_idx_to_del.append(idx)
        org_dtype = data.dtype
        data = data.to(torch.float32).numpy() if org_dtype == torch.bfloat16 else data.numpy()
        # Split on axis 0 for enabling transpose_b
        w_left, w_right = np.split(data, [ffn_hidden_size], axis=0)
        left_name = name.replace("dense_h_to_4h", "dense_left")
        right_name = name.replace("dense_h_to_4h", "dense_right")
        param_split.append({"name": left_name, "data": npy2ms(w_left, dtype)})
        param_split.append({"name": right_name, "data": npy2ms(w_right, dtype)})

    for idx in reversed(param_idx_to_del):  # delete element in list by index
        param_list.pop(idx)

    param_list += param_split


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, config, dtype=ms.float32, **kwargs):
    """ Convert pytorch model file to MindSpore model file. """
    config: ChatGLM2Config = MindFormerConfig(config)['model']['model_config']
    config = ChatGLM2Config(**config)
    model = AutoModel.from_pretrained(input_path)

    print('parameter convert....')
    ms_param = []
    for k, v in tqdm(model.state_dict().items()):
        if "word_embeddings.weight" in k:
            k = k.replace("word_embeddings.weight", "embedding_weight")
        ms_param.append({"name": k, "data": v})
    # qkv weight split
    if not config.qkv_concat or config.use_rearrange_rope:
        attn_split(ms_param, config, dtype)

    # mlp weight split
    if not config.mlp_concat:
        mlp_split(ms_param, config, dtype)

    tmp_list = []
    pop_list = []
    for i, item in enumerate(ms_param):
        k, v = item["name"], item["data"]
        if not isinstance(v, ms.Tensor):
            tmp_list.append({"name": k, "data": pt2ms(v, dtype)})
            pop_list.append(i)
    for i in reversed(pop_list):
        ms_param.pop(i)
    ms_param += tmp_list

    ms.save_checkpoint(ms_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


def convert_to_concat_ckpt(ms_not_concat_ckpt_path, ms_concat_ckpt_path, config_path):
    """convert previous ckpt to concat ckpt"""
    model_config = MindFormerConfig(config_path).model
    if 'auto_register' in model_config:
        MindFormerRegister.auto_register(class_reference=model_config.pop('auto_register'),
                                         module_type=MindFormerModuleType.MODELS)

    if os.path.isdir(ms_not_concat_ckpt_path):
        rank_dir_list = os.listdir(ms_not_concat_ckpt_path)
        for rank_dir in rank_dir_list:
            rank_dir_name = str(rank_dir)
            rank_id = rank_dir_name.split("_")[1]
            checkpoint_path = os.path.join(ms_not_concat_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            print("checkpoint_path: {}".format(checkpoint_path))
            params = ms.load_checkpoint(checkpoint_path)
            params = concat_weight_and_bias(params, model_config)

            save_dir = os.path.join(ms_concat_ckpt_path, rank_dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(ms_concat_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            ms.save_checkpoint(params, save_path)
    else:
        params = ms.load_checkpoint(ms_not_concat_ckpt_path)
        params = concat_weight_and_bias(params, model_config)
        ms.save_checkpoint(params, ms_concat_ckpt_path)


def concat_param(param_dict, param_name_list, concat_name):
    param_value_list = list()
    for param_name in param_name_list:
        param_value_list.append(param_dict[param_name].asnumpy())
        param_dict.pop(param_name)
    concat_value = np.concatenate(param_value_list, 0)
    param_dict[concat_name] = ms.Parameter(concat_value, name=concat_name)
    print("transform: {}".format(concat_name))


def concat_weight_and_bias(param_dict, config):
    """convert qkv concat weight"""
    qkv_weight_name = "transformer.encoder.layers.{}.self_attention.{}.weight"
    qkv_bias_name = "transformer.encoder.layers.{}.self_attention.{}.bias"
    mlp_weight_name = "transformer.encoder.layers.{}.mlp.{}.weight"
    for i in range(config.model_config.num_layers):
        # qkv weight concat
        qkv_weight_param_name_list = [qkv_weight_name.format(i, "wq"),
                                      qkv_weight_name.format(i, "wk"),
                                      qkv_weight_name.format(i, "wv")]
        qkv_weight_concat_name = qkv_weight_name.format(i, "query_key_value")
        concat_param(param_dict=param_dict,
                     param_name_list=qkv_weight_param_name_list,
                     concat_name=qkv_weight_concat_name)

        mlp_weight_param_name_list = [mlp_weight_name.format(i, "dense_left"),
                                      mlp_weight_name.format(i, "dense_right")]
        mlp_weight_concat_name = mlp_weight_name.format(i, "dense_h_to_4h")
        concat_param(param_dict=param_dict,
                     param_name_list=mlp_weight_param_name_list,
                     concat_name=mlp_weight_concat_name)

        if config.model_config.add_qkv_bias:
            qkv_bias_param_name_list = [qkv_bias_name.format(i, "wq"),
                                        qkv_bias_name.format(i, "wk"),
                                        qkv_bias_name.format(i, "wv")]
            qkv_bias_concat_name = qkv_bias_name.format(i, "query_key_value")
            concat_param(param_dict=param_dict,
                         param_name_list=qkv_bias_param_name_list,
                         concat_name=qkv_bias_concat_name)
    return param_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLM2/3 weight convert script")
    parser.add_argument("--torch_ckpt_path",
                        type=str,
                        default="None",
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help='The output mindspore checkpoint path.')
    parser.add_argument("--dtype",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="The dtype of transformed mindspore weight.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to model config yaml")
    parser.add_argument('--concat', default=False, type=str2bool, help="Whether to concat weight and bias")
    parser.add_argument('--ms_not_concat_ckpt_path', default=None)
    mapping = {
        "fp32": ms.float32,
        "fp16": ms.float16,
        "bf16": ms.bfloat16
    }

    opt = parser.parse_args()
    if opt.config is None:
        raise RuntimeError("config must be specified")
    if opt.concat:
        convert_to_concat_ckpt(ms_not_concat_ckpt_path=opt.ms_not_concat_ckpt_path,
                               ms_concat_ckpt_path=opt.mindspore_ckpt_path,
                               config_path=opt.config)
    else:
        convert_pt_to_ms(input_path=opt.torch_ckpt_path, output_path=opt.mindspore_ckpt_path,
                         dtype=mapping.get(opt.dtype, ms.bfloat16),
                         config=opt.config)
