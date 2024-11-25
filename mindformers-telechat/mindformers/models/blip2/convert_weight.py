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

""" Convert checkpoint from salesforce."""
from collections import OrderedDict
import argparse

import mindspore as ms
import mindspore.ops as P
import torch

from mindformers.utils.convert_utils import pt2ms


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    r"""Convert Weight
    Convert blip2_qformer weights from pytorch to mindspore,
    pytorch (CPU) required.

    Args:
        input_path: The torch path to blip2_qformer.
        output_path: The save path for  blip2_qformer.
        dtype: dtype.

    Returns:
        the converted mindspore_model_weight for Blip2Qformer class.

    Notice: when loading, warning of: ""2 parameters 'visual_encoder.fc_norm.gamma',
             'visual_encoder.fc_norm.beta' ara not loaded. " is normal,
             because bilp2 utilizes the penultimate layer output of the eva_vit.
    """
    pt_params = torch.load(input_path, map_location='cpu')
    if not isinstance(pt_params, OrderedDict):
        if isinstance(pt_params, dict) and 'model' in pt_params.keys():
            pt_params = pt_params['model']
        else:
            raise ValueError("wrong torch state_dict format when loading {},\
                please check.".format(input_path))

    name_pt2ms = {
        "cls_token": "cls_tokens",
        "attn.proj": "attention.projection",
        "attn.q_bias": "attention.dense1.bias",
        "attn.v_bias": "attention.dense3.bias",
        "norm1.weight": "layernorm1.gamma",
        "norm1.bias": "layernorm1.beta",
        "norm2.weight": "layernorm2.gamma",
        "norm2.bias": "layernorm2.beta",
        "fc_norm.weight": "fc_norm.gamma",
        "fc_norm.bias": "fc_norm.beta",
        "ln_vision.weight": "ln_vision.gamma",
        "ln_vision.bias": "ln_vision.beta",
        "mlp.fc2.weight": "output.mapping.weight",
        "mlp.fc1.bias": "output.mapping.bias",
        "mlp.fc1.weight": "output.projection.weight",
        "mlp.fc2.bias": "output.projection.bias",
        "LayerNorm.": "layernorm.",
        "layernorm.weight": "layernorm.gamma",
        "layernorm.bias": "layernorm.beta",
        "embeddings.weight": "embeddings.embedding_table",
        "self": "self_att",
    }
    ms_param_dict = []
    for pt_name, pt_tensor in pt_params.items():
        # initial name assign
        ms_name = pt_name
        # extract data
        data = pt2ms(pt_tensor, dtype)
        # split qkv weights
        if "qkv.weight" in pt_name:
            length = pt_tensor.shape[0] // 3
            ms_name1 = pt_name.replace("attn.qkv", "attention.dense1")
            ms_name2 = pt_name.replace("attn.qkv", "attention.dense2")
            ms_name3 = pt_name.replace("attn.qkv", "attention.dense3")
            ms_param_dict.append({"name": ms_name1, "data": data[:length]})
            ms_param_dict.append({"name": ms_name2, "data": data[length:length * 2]})
            ms_param_dict.append({"name": ms_name3, "data": data[length * 2:length * 3]})
            print("rename {} to {}, {} and {}".format(pt_name, ms_name1, ms_name2, ms_name3))
        else:
            #  Rename
            for replace_from, replace_to in name_pt2ms.items():
                ms_name = ms_name.replace(replace_from, replace_to)
                ms_name = ms_name.replace("Qformer.", "qformer.")
            if ms_name.endswith("output.mapping.weight") or \
                    ms_name.endswith("output.projection.weight") or \
                    ms_name.endswith("attention.projection.weight"):
                data = data.T
            ms_param_dict.append({"name": ms_name, "data": data})
            # when loading each query-bias, append a zero value key-bias.
            if ms_name.endswith("attention.dense1.bias"):
                ms_param_dict.append({
                    "name": ms_name.replace("dense1", "dense2"),
                    "data": P.zeros_like(data)
                })
            if ms_name != pt_name:
                print(f"rename {pt_name} to {ms_name}")

    print(f"\n----------------- convert {input_path} to {output_path} Finished! -----------------\n")
    ms.save_checkpoint(ms_param_dict, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="blip2 weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="blip2_pretrain.pth",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="blip2_pretrain.ckpt",
                        help="The output mindspore checkpoint path.")
    opt = parser.parse_args()

    convert_pt_to_ms(opt.torch_path, opt.mindspore_path)
