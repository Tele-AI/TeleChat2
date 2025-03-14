# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import argparse
from collections import OrderedDict
import torch

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

ms.set_context(device_id=1)


def convert_vit_weight(pt_weight_params, vit_mindspore_path):
    """
    convert vit weight from torch
    """
    vit_name_pt2ms = {
        "mixins.eva.model.vit.transformer.word_embeddings.weight": "visual_encoder.cls_tokens",
        "mixins.eva.model.vit.transformer.position_embeddings.weight": "visual_encoder.pos_embed",
        "mixins.eva.model.vit.mixins.patch_embedding.proj.weight": "visual_encoder.patch_embed.proj.weight",
        "mixins.eva.model.vit.mixins.patch_embedding.proj.bias": "visual_encoder.patch_embed.proj.bias",
        "mixins.eva.model.vit.mixins.cls.ln_vision.weight": "ln_vision.gamma",
        "mixins.eva.model.vit.mixins.cls.ln_vision.bias": "ln_vision.beta",
    }
    vit_name_pt2ms_reg = {
        "mixins.eva.model.vit.mixins": "visual_encoder",
        "mixins.eva.model.vit.transformer": "visual_encoder",
        "layers": "blocks",
        "input_layernorm.weight": "layernorm1.gamma",
        "input_layernorm.bias": "layernorm1.beta",
        "post_attention_layernorm.weight": "layernorm2.gamma",
        "post_attention_layernorm.bias": "layernorm2.beta",
        "attention.dense.weight": "attention.projection.weight",
        "attention.dense.bias": "attention.projection.bias",
        "mlp.dense_h_to_4h.weight": "output.mapping.weight",
        "mlp.dense_h_to_4h.bias": "output.mapping.bias",
        "mlp.dense_4h_to_h.weight": "output.projection.weight",
        "mlp.dense_4h_to_h.bias": "output.projection.bias"
    }

    ms_param_dict = []
    for pt_name, pt_tensor in pt_weight_params.items():
        ms_name = pt_name
        numpy_value = pt_weight_params[pt_name].detach().numpy()
        pt_dtype = str(pt_tensor.dtype)
        if pt_dtype == "torch.float16":
            ms_dtype = mstype.float16
        else:
            ms_dtype = mstype.float32

        data = Tensor(numpy_value, dtype=ms_dtype)

        # replace vit related params
        if "vit" in ms_name:
            for replace_from, replace_to in vit_name_pt2ms.items():
                if ms_name == replace_from:
                    ms_name = replace_to

            for replace_from, replace_to in vit_name_pt2ms_reg.items():
                ms_name = ms_name.replace(replace_from, replace_to)

            if "attention.query_key_value" in ms_name:
                length = data.shape[0] // 3
                ms_name1 = ms_name.replace("query_key_value", "dense1")
                ms_name2 = ms_name.replace("query_key_value", "dense2")
                ms_name3 = ms_name.replace("query_key_value", "dense3")
                ms_param_dict.append({"name": ms_name1, "data": data[:length]})
                ms_param_dict.append({"name": ms_name2, "data": data[length:length * 2]})
                ms_param_dict.append({"name": ms_name3, "data": data[length * 2:length * 3]})
                print(f"rename {pt_name} to {ms_name1}, {ms_name2} and {ms_name3} with type {data.dtype}")
            elif "cls_tokens" in ms_name or "pos_embed" in ms_name:
                ms_param_dict.append({"name": ms_name, "data": data.unsqueeze(0)})
                print(f"convert {pt_name} to {ms_name} with shape {data.unsqueeze(0).shape} and type {data.dtype}")
            elif "output.mapping.weight" in ms_name or "output.projection.weight" in ms_name or \
                    "attention.projection.weight" in ms_name:
                ms_param_dict.append({"name": ms_name, "data": data.T})
                print(f"convert {pt_name} to {ms_name} with shape {data.T.shape} and type {data.dtype}")
            else:
                ms_param_dict.append({"name": ms_name, "data": data})
                print(f"convert {pt_name} to {ms_name} with shape {data.shape} and type {data.dtype}")

        if "ln_vision" in ms_name:
            if "weight" in ms_name:
                ms_param_dict.append({"name": "ln_vision.gamma", "data": data})
            else:
                ms_param_dict.append({"name": "ln_vision.beta", "data": data})

    print("\n----------------- convert vit pytorch model to mindspore model Finished! -----------------\n")
    ms.save_checkpoint(ms_param_dict, vit_mindspore_path)


def convert_glm_weight(pt_weight_params, glm_mindspore_path):
    """
    convert glm weight from torch
    """
    num_layers = 28
    print('chatglm parameter convert....')
    ms_param = []
    ms_param_lite = []
    for pt_name, pt_value in pt_weight_params.items():
        print('current parameter: ', pt_name)
        if 'mixins.eva' in pt_name:
            continue
        if pt_name != 'mixins.chatglm-attn.rotary_emb.inv_freq':
            if "transformer.word_embeddings.weight" in pt_name or "transformer.position_embeddings.weight" in pt_name:
                pt_name = pt_name.replace("weight", "embedding_table")
                ms_param_lite.append({"name": pt_name, "data": ms.Tensor(pt_value.numpy())})
            if "post_attention_layernorm" in pt_name or "input_layernorm" in pt_name or "final_layernorm" in pt_name:
                pt_name = pt_name.replace("weight", "gamma")
                pt_name = pt_name.replace("bias", "beta")
            if "mixins.chatglm-final.lm_head" in pt_name:
                pt_name = pt_name.replace("mixins.chatglm-final.lm_head", "lm_head")
            ms_param.append({"name": pt_name, "data": ms.Tensor(pt_value.numpy())})
        else:
            for layer_id in range(num_layers):
                pt_name = f"transformer.layers.{layer_id}.attention.rotary_emb.inv_freq"
                ms_param.append({"name": pt_name, "data": ms.Tensor(pt_value.numpy())})

        if "ln_vision" in pt_name:
            if "weight" in pt_name:
                ms_param.append({"name": "ln_vision.gamma", "data": ms.Tensor(pt_value.numpy())})
            else:
                ms_param.append({"name": "ln_vision.beta", "data": ms.Tensor(pt_value.numpy())})

        if "glm_proj" in pt_name:
            if "weight" in pt_name:
                ms_param.append({"name": "llm_proj.weight", "data": ms.Tensor(pt_value.numpy())})
            else:
                ms_param.append({"name": "llm_proj.bias", "data": ms.Tensor(pt_value.numpy())})

    print('saving ms ckpt....')

    glm_for_lite_path = os.path.join(os.path.dirname(glm_mindspore_path), 'glm_6b_for_lite.ckpt')
    ms.save_checkpoint(ms_param_lite, glm_for_lite_path)
    ms.save_checkpoint(ms_param, glm_mindspore_path)



def convert_qformer_weight(pt_weight_params, qformer_mindspore_path):
    """
    convert qformer weight from torch
    """
    qformer_name_convert_reg = {
        "mixins.eva.model.qformer.transformer.final_layernorm.weight": "qformer.bert.encoder.final_layernorm.gamma",
        "mixins.eva.model.qformer.transformer.final_layernorm.bias": "qformer.bert.encoder.final_layernorm.beta",
        "mixins.eva.model.qformer.transformer.layers.": "qformer.bert.encoder.layer.",
        ".attention.dense.": ".attention.output.dense.",
        ".input_layernorm.weight": ".input_layernorm.gamma",
        ".input_layernorm.bias": ".input_layernorm.beta",
        ".post_attention_layernorm.weight": ".attention.output.layernorm.gamma",
        ".post_attention_layernorm.bias": ".attention.output.layernorm.beta",
        ".cross_attention.dense.": ".crossattention.output.dense.",
        ".cross_attention.": ".crossattention.",
        ".post_cross_attention_layernorm.weight": ".crossattention.output.layernorm.gamma",
        ".post_cross_attention_layernorm.bias": ".crossattention.output.layernorm.beta",
        ".mlp.dense_h_to_4h.": ".intermediate_query.dense.",
        ".mlp.dense_4h_to_h.": ".output_query.dense.",
        ".query.": ".self_att.query."
    }

    ms_param_dict = []
    for pt_name, pt_tensor in pt_weight_params.items():
        ms_name = pt_name
        numpy_value = pt_weight_params[pt_name].detach().numpy()
        pt_dtype = str(pt_tensor.dtype)
        if pt_dtype == "torch.float16":
            ms_dtype = mstype.float16
        else:
            ms_dtype = mstype.float32

        data = Tensor(numpy_value, dtype=ms_dtype)

        # replace qformer related params
        if "qformer" in ms_name:
            for replace_from, replace_to in qformer_name_convert_reg.items():
                ms_name = ms_name.replace(replace_from, replace_to)

            if "query_key_value" in ms_name:
                length = data.shape[0] // 3
                ms_name_query = ms_name.replace("query_key_value", "self_att.query")
                ms_name_key = ms_name.replace("query_key_value", "self_att.key")
                ms_name_value = ms_name.replace("query_key_value", "self_att.value")
                ms_param_dict.append({"name": ms_name_query, "data": data[:length]})
                ms_param_dict.append({"name": ms_name_key, "data": data[length:length * 2]})
                ms_param_dict.append({"name": ms_name_value, "data": data[length * 2:length * 3]})
                print(
                    f"rename {pt_name} to {ms_name_query}, {ms_name_key} and {ms_name_value} with type {data.dtype}")
            elif "key_value" in ms_name:
                length = data.shape[0] // 2
                ms_name_key = ms_name.replace("key_value", "self_att.key")
                ms_name_value = ms_name.replace("key_value", "self_att.value")
                ms_param_dict.append({"name": ms_name_key, "data": data[:length]})
                ms_param_dict.append({"name": ms_name_value, "data": data[length:length * 2]})
                print(f"rename {pt_name} to {ms_name_key} and {ms_name_value} with type {data.dtype}")

            elif ms_name == "mixins.eva.model.qformer.transformer.word_embeddings.weight":
                ms_name = "query_tokens"
                shape = data.shape
                data = data.reshape((1, shape[0], shape[1]))
                ms_param_dict.append({"name": ms_name, "data": data})
                print(f"convert {pt_name} to {ms_name} with shape {data.shape} and type {data.dtype}")
            else:
                ms_param_dict.append({"name": ms_name, "data": data})
                print(f"convert {pt_name} to {ms_name} with shape {data.shape} and type {data.dtype}")

        if "ln_vision" in ms_name:
            if "weight" in ms_name:
                ms_param_dict.append({"name": "ln_vision.gamma", "data": data})
                print(f"convert {pt_name} to ln_vision.gamma with shape {data.shape} and type {data.dtype}")
            else:
                ms_param_dict.append({"name": "ln_vision.beta", "data": data})
                print(f"convert {pt_name} to ln_vision.beta with shape {data.shape} and type {data.dtype}")

        if "glm_proj" in ms_name:
            if "weight" in ms_name:
                ms_param_dict.append({"name": "llm_proj.weight", "data": data})
                print(f"convert {pt_name} to llm_proj.weight with shape {data.shape} and type {data.dtype}")
            else:
                ms_param_dict.append({"name": "llm_proj.bias", "data": data})
                print(f"convert {pt_name} to llm_proj.bias with shape {data.shape} and type {data.dtype}")

    print('saving qformer ckpt....')
    ms.save_checkpoint(ms_param_dict, qformer_mindspore_path)


def convert_weight(args):
    r"""Convert Weight
    Convert visualglm weights from pytorch to mindspore,
    pytorch (CPU) required.

    Args:
        args: The input parameters for convertting torch model to mindspore model.

    Returns:
        the converted mindspore_model_weight for visualglm class.
    """
    pt_params = torch.load(args.torch_path, map_location='cpu')
    if not isinstance(pt_params, OrderedDict):
        if isinstance(pt_params, dict) and 'module' in pt_params.keys():
            pt_params = pt_params['module']
        else:
            raise ValueError(f"wrong torch state_dict format when loading {args.torch_path}, please check.")

    if args.vit_convert_flag:
        convert_vit_weight(pt_params, args.vit_mindspore_path)

    if args.qformer_convert_flag:
        convert_qformer_weight(pt_params, args.qformer_mindspore_path)

    if args.glm_convert_flag:
        convert_glm_weight(pt_params, args.glm_mindspore_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="blip2 weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="/opt/visualglm/model_sat/visualglm-6b/1/mp_rank_00_model_states.pt",
                        help="The torch checkpoint path.")
    parser.add_argument("--vit_mindspore_path",
                        type=str,
                        default="/dev/zsh/models/visualglm/visualglm_vit.ckpt",
                        help="The output mindspore vit model checkpoint path.")
    parser.add_argument("--qformer_mindspore_path",
                        type=str,
                        default="/dev/zsh/models/visualglm/visualglm_qformer.ckpt",
                        help="The output mindspore qformer model checkpoint path.")
    parser.add_argument("--glm_mindspore_path",
                        type=str,
                        default="/dev/zsh/models/visualglm/glm_6b.ckpt",
                        help="The output mindspore glm model checkpoint path.")
    parser.add_argument("--vit_convert_flag",
                        type=int,
                        default=1,
                        help="whether the vit model needs to be converted")
    parser.add_argument("--qformer_convert_flag",
                        type=int,
                        default=1,
                        help="whether the qformer model needs to be converted")
    parser.add_argument("--glm_convert_flag",
                        type=int,
                        default=1,
                        help="whether the glm model needs to be converted")

    opt = parser.parse_args()

    convert_weight(opt)
