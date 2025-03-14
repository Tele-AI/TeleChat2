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
import argparse
import json
import os
from collections import defaultdict
from glob import glob
import warnings

import mindspore as ms
import torch
from safetensors.torch import load_file


dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}

default_config = {
    'num_routed_experts': 256,
    'n_head': 128,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'num_layers': 61,
    'num_nextn_predict_layers': 1,
    'first_k_dense_replace': 3,
    'dtype': ms.bfloat16,
    'use_gemm': False,
    'save_format': "safetensors"
}


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape (M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    # Get the original dimensions of weight
    m, n = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    if scale_m != (m + block_size - 1) // block_size:
        raise ValueError("Mismatch in scale rows and weight rows.")
    if scale_n != (n + block_size - 1) // block_size:
        raise ValueError("Mismatch in scale columns and weight columns.")

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:m, :n]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


def dequant_layer_weights(layer_id, pt_layer_weights):
    """Dequanting weights in a layer"""
    dequanted_weights = {}
    for weight_name, weight in pt_layer_weights.items():
        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1 and (f"model.layers.{layer_id}." in weight_name):  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                # Get scale_inv from the correct file
                scale_inv = pt_layer_weights.get(scale_inv_name)
                dequanted_weights[weight_name] = weight_dequant(weight, scale_inv)
            except KeyError:
                print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping dequanting")
                dequanted_weights[weight_name] = weight
        else:
            dequanted_weights[weight_name] = weight
    return dequanted_weights


def plain_name_replace(weight_name: str):
    """Weight name replacing for pre/post-process module"""
    weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
    return weight_name


def mla_name_replace(weight_name: str):
    """Weight name replacing for MLA module weights"""
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.q_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
    weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
    weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
    weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
    weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


def mlp_name_replace(weight_name: str):
    """Weight name replacing for MLP module, including MoE"""
    weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
    weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
    weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
    weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
    weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                      'feed_forward.routed_experts.router.router.topk_bias')
    return weight_name


def mtp_name_replace(weight_name: str, current_layer_id: int, mtp_layer_id: int):
    """replace weight name for MultiPredictionToken module"""
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.enorm",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.norm_emb")
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.hnorm",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.norm")
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.eh_proj",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.dense")
    return weight_name


def load_data_pt(file_name):
    return load_file(file_name, device="cpu")


def layers_model_file_map(file_path):
    """Get weight-file map"""
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "model.safetensors.index.json")
    if os.path.exists(weight_map_file):
        with open(weight_map_file) as f:
            weights_map = json.load(f)
        weights_map = weights_map["weight_map"]
    else:
        warnings.warn(f"Cannot find weight map file model.safetensors.index.json in path {file_path}, " \
                    f"Trying to load one safetensor file ...")
        files = sorted(glob(os.path.join(file_path, "*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors files found in path {file_path}")

        weight_file = files[0].split("/")[-1]
        keys = load_data_pt(os.path.join(file_path, weight_file)).keys()
        weights_map = {}
        for k in keys:
            weights_map[k] = weight_file

    for weight_key, value in weights_map.items():
        if weight_key.startswith("model.layers."):
            layer_name = int(weight_key.split('model.layers.')[1].split('.')[0])
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def read_matched_file_pt(layer_st_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_first:
        st_file_list.extend(list(layer_st_map["model.embed_tokens.weight"]))
    if is_last:
        st_file_list.extend(list(layer_st_map["model.norm.weight"]))
        st_file_list.extend(list(layer_st_map["lm_head.weight"]))
    st_file_list = list(set(st_file_list))
    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_pt(st_file)
        weights.update(current_weight)
    return weights


def _mla_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MLA module"""
    n_head = config['n_head']
    qk_nope_head_dim = config['qk_nope_head_dim']
    qk_rope_head_dim = config['qk_rope_head_dim']
    v_head_dim = config['v_head_dim']

    q_a_proj_key = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
    kv_a_proj_key = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
    o_proj_key = f"model.layers.{layer_id}.self_attn.o_proj.weight"
    q_a_layernorm_key = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
    kv_a_layernorm_key = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
    q_b_proj_key = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
    kv_b_proj_key = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
    input_norm_key = f"model.layers.{layer_id}.input_layernorm.weight"
    post_attn_norm_key = f"model.layers.{layer_id}.post_attention_layernorm.weight"

    q_a_proj = pt_layer_weights.pop(q_a_proj_key)
    kv_a_proj = pt_layer_weights.pop(kv_a_proj_key)
    o_proj = pt_layer_weights.pop(o_proj_key)
    q_a_layernorm = pt_layer_weights.pop(q_a_layernorm_key)
    kv_a_layernorm = pt_layer_weights.pop(kv_a_layernorm_key)
    q_b_proj = pt_layer_weights.pop(q_b_proj_key)
    kv_b_proj = pt_layer_weights.pop(kv_b_proj_key)
    input_norm = pt_layer_weights.pop(input_norm_key)
    post_attn_norm = pt_layer_weights.pop(post_attn_norm_key)

    mla_weight_dict = defaultdict()
    # split q_b_proj
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    qk_nope, qk_rope = q_b_proj.reshape(n_head, qk_head_dim, -1).split([qk_nope_head_dim, qk_rope_head_dim], dim=1)
    qk_rope = qk_rope.reshape(qk_rope.shape[0], qk_rope.shape[1] // 2, 2, -1).permute(0, 2, 1, 3)
    qk_nope = qk_nope.reshape(-1, qk_nope.shape[-1])
    qk_rope = qk_rope.reshape(-1, qk_rope.shape[-1])
    qk_nope_key = mla_name_replace(q_b_proj_key).replace(".l2q_proj.", ".l2q_nope_proj.")
    qk_rope_key = mla_name_replace(q_b_proj_key).replace(".l2q_proj.", ".l2q_pe_proj.")
    mla_weight_dict[qk_nope_key] = qk_nope.clone()
    mla_weight_dict[qk_rope_key] = qk_rope.clone()
    # split kv_a_proj
    kv_lora_rank = kv_a_proj.shape[0] - qk_rope_head_dim
    latent_kv, k_rope = kv_a_proj.split([kv_lora_rank, qk_rope_head_dim], dim=0)
    k_rope = k_rope.reshape(k_rope.shape[0] // 2, 2, -1).permute(1, 0, 2).reshape(-1, k_rope.shape[-1])
    latent_kv_key = mla_name_replace(kv_a_proj_key).replace(".kv2l.", ".kv2l_latent_kv.")
    k_rope_key = mla_name_replace(kv_a_proj_key).replace(".kv2l.", ".kv2l_k_pe.")
    mla_weight_dict[latent_kv_key] = latent_kv.clone()
    mla_weight_dict[k_rope_key] = k_rope.clone()
    # split kv_b_proj
    kv_head_dim = qk_nope_head_dim + v_head_dim
    k_nope, v = kv_b_proj.reshape(n_head, kv_head_dim, -1).split([qk_nope_head_dim, v_head_dim], dim=1)
    k_nope = k_nope.reshape(-1, k_nope.shape[-1])
    v = v.reshape(-1, v.shape[-1])
    k_nope_key = mla_name_replace(kv_b_proj_key).replace(".lkv2kv.", ".lkv2kv_k_nope.")
    v_key = mla_name_replace(kv_b_proj_key).replace(".lkv2kv.", ".lkv2kv_v.")
    mla_weight_dict[k_nope_key] = k_nope
    mla_weight_dict[v_key] = v
    # process q_a_proj, o_proj, and layernorms
    q_a_proj_key = mla_name_replace(q_a_proj_key)
    mla_weight_dict[q_a_proj_key] = q_a_proj.clone()
    o_proj_key = mla_name_replace(o_proj_key)
    mla_weight_dict[o_proj_key] = o_proj.clone()
    q_a_layernorm_key = mla_name_replace(q_a_layernorm_key)
    mla_weight_dict[q_a_layernorm_key] = q_a_layernorm.clone()
    kv_a_layernorm_key = mla_name_replace(kv_a_layernorm_key)
    mla_weight_dict[kv_a_layernorm_key] = kv_a_layernorm.clone()
    input_norm_key = mla_name_replace(input_norm_key)
    mla_weight_dict[input_norm_key] = input_norm.clone()
    post_attn_norm_key = mla_name_replace(post_attn_norm_key)
    mla_weight_dict[post_attn_norm_key] = post_attn_norm.clone()

    return mla_weight_dict


def _mlp_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MLP/MoE module"""
    num_routed_experts = config['num_routed_experts']
    first_k_dense_replace = config['first_k_dense_replace']
    use_gemm = config['use_gemm']

    mlp_weight_dict = defaultdict()
    if layer_id < first_k_dense_replace:
        gate_proj_key = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        up_proj_key = f"model.layers.{layer_id}.mlp.up_proj.weight"
        down_proj_key = f"model.layers.{layer_id}.mlp.down_proj.weight"
        gate_proj = pt_layer_weights.pop(gate_proj_key)
        up_proj = pt_layer_weights.pop(up_proj_key)
        down_proj = pt_layer_weights.pop(down_proj_key)

        gate_proj_key = mlp_name_replace(gate_proj_key)
        up_proj_key = mlp_name_replace(up_proj_key)
        down_proj_key = mlp_name_replace(down_proj_key)
        mlp_weight_dict[gate_proj_key] = gate_proj.clone()
        mlp_weight_dict[up_proj_key] = up_proj.clone()
        mlp_weight_dict[down_proj_key] = down_proj.clone()
    else:
        router_weight_key = f"model.layers.{layer_id}.mlp.gate.weight"
        router_correct_bias_key = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        shared_experts_gate_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        shared_experts_up_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
        shared_experts_down_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        router_weight = pt_layer_weights.pop(router_weight_key)
        router_weight = router_weight[:num_routed_experts, :]
        router_correct_bias = pt_layer_weights.pop(router_correct_bias_key)
        router_correct_bias = router_correct_bias[:num_routed_experts]
        shared_experts_gate_proj = pt_layer_weights.pop(shared_experts_gate_proj_key)
        shared_experts_up_proj = pt_layer_weights.pop(shared_experts_up_proj_key)
        shared_experts_down_proj = pt_layer_weights.pop(shared_experts_down_proj_key)

        gate_proj_list = []
        up_proj_list = []
        down_proj_list = []
        for expert_id in range(num_routed_experts):
            gate_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
            up_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
            down_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
            gate_proj = pt_layer_weights.pop(gate_proj_key)
            up_proj = pt_layer_weights.pop(up_proj_key)
            down_proj = pt_layer_weights.pop(down_proj_key)
            gate_proj_list.append(gate_proj)
            up_proj_list.append(up_proj)
            down_proj_list.append(down_proj)

        expert_gate_proj = torch.stack(gate_proj_list, 0)
        expert_up_proj = torch.stack(up_proj_list, 0)
        expert_down_proj = torch.stack(down_proj_list, 0)

        # replace name and store
        router_weight_key = mlp_name_replace(router_weight_key)
        router_correct_bias_key = mlp_name_replace(router_correct_bias_key)
        shared_experts_gate_proj_key = mlp_name_replace(shared_experts_gate_proj_key)
        shared_experts_up_proj_key = mlp_name_replace(shared_experts_up_proj_key)
        shared_experts_down_proj_key = mlp_name_replace(shared_experts_down_proj_key)
        mlp_weight_dict[router_weight_key] = router_weight.clone()
        mlp_weight_dict[router_correct_bias_key] = router_correct_bias.clone()
        mlp_weight_dict[shared_experts_gate_proj_key] = shared_experts_gate_proj.clone()
        mlp_weight_dict[shared_experts_up_proj_key] = shared_experts_up_proj.clone()
        mlp_weight_dict[shared_experts_down_proj_key] = shared_experts_down_proj.clone()
        # routed experts
        expert_gate_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        expert_up_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"
        expert_down_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        if use_gemm:
            expert_gate_proj = expert_gate_proj.transpose(1, 2)
            expert_up_proj = expert_up_proj.transpose(1, 2)
            expert_down_proj = expert_down_proj.transpose(1, 2)
        mlp_weight_dict[expert_gate_proj_key] = expert_gate_proj.clone()
        mlp_weight_dict[expert_up_proj_key] = expert_up_proj.clone()
        mlp_weight_dict[expert_down_proj_key] = expert_down_proj.clone()

    return mlp_weight_dict


def _mtp_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MTP module, the shared weights will be ignored"""
    num_layers = config["num_layers"]
    mtp_layer_id = layer_id - num_layers
    # ignore the shared emb_weights and lm head in mtp layers
    pt_layer_weights.pop(f"model.layers.{layer_id}.embed_tokens.weight")
    pt_layer_weights.pop(f"model.layers.{layer_id}.shared_head.norm.weight")
    pt_layer_weights.pop(f"model.layers.{layer_id}.shared_head.head.weight")
    enorm_key = f"model.layers.{layer_id}.enorm.weight"
    hnorm_key = f"model.layers.{layer_id}.hnorm.weight"
    e_proj_key = f"model.layers.{layer_id}.eh_proj.weight"

    enorm = pt_layer_weights.pop(enorm_key)
    hnorm = pt_layer_weights.pop(hnorm_key)
    e_proj = pt_layer_weights.pop(e_proj_key)

    mtp_weight_dict = defaultdict()
    enorm_key = mtp_name_replace(enorm_key, layer_id, mtp_layer_id)
    hnorm_key = mtp_name_replace(hnorm_key, layer_id, mtp_layer_id)
    e_proj_key = mtp_name_replace(e_proj_key, layer_id, mtp_layer_id)
    mtp_weight_dict[enorm_key] = enorm.clone()
    mtp_weight_dict[hnorm_key] = hnorm.clone()
    mtp_weight_dict[e_proj_key] = e_proj.clone()

    return mtp_weight_dict


def _model_preprocess_pt_to_ms(pt_layer_weights):
    """Processing weights in prepross module"""
    emb_weight_key = "model.embed_tokens.weight"
    emb_weight = pt_layer_weights.pop(emb_weight_key)
    emb_weight_key = plain_name_replace(emb_weight_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = emb_weight.clone()

    return plain_weight_dict


def _model_postprocess_pt_to_ms(pt_layer_weights):
    """Processing weights in postpross module"""
    final_norm_key = "model.norm.weight"
    lm_head_key = "lm_head.weight"
    final_norm = pt_layer_weights.pop(final_norm_key)
    lm_head = pt_layer_weights.pop(lm_head_key)

    final_norm_key = plain_name_replace(final_norm_key)
    lm_head_key = plain_name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[final_norm_key] = final_norm.clone()
    plain_weight_dict[lm_head_key] = lm_head.clone()

    return plain_weight_dict


def ms_ckpt_convertor(input_path, output_path, config):
    """Convert to ckpt format checkpoint"""
    # check output_path
    if output_path.endswith(".ckpt"):
        saving_file = output_path
    else:
        saving_file = os.path.join(output_path, "checkpoints.ckpt")

    layer_st_map = layers_model_file_map(input_path)
    torch.set_default_dtype(torch.bfloat16)

    dtype = config["dtype"]
    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    ms_weights = defaultdict()
    for layer_id in range(total_num_layers):
        if layer_id == 0:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif layer_id == total_num_layers - 1:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=True)
        else:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=False)
        # first dequanting weights
        pt_layer_weights = dequant_layer_weights(layer_id, pt_layer_weights)

        if layer_id == 0:
            ms_weights.update(_model_preprocess_pt_to_ms(pt_layer_weights))
        ms_weights.update(_mla_pt_to_ms(layer_id, pt_layer_weights, config))
        ms_weights.update(_mlp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id > num_layers - 1:
            ms_weights.update(_mtp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id == total_num_layers - 1:
            ms_weights.update(_model_postprocess_pt_to_ms(pt_layer_weights))

    to_save_ckpt = []
    for name in list(ms_weights.keys()):
        value = ms_weights.pop(name).to(torch.float32).numpy()
        tmp_dtype = dtype
        if "norm" in name or "router.dense" in name or "topk_bias" in name:
            tmp_dtype = ms.float32
        to_save_ckpt.append({'name': name, 'data': ms.Tensor(value, dtype=tmp_dtype)})

    print(f"Saving weights to file {saving_file}")
    ms.save_checkpoint(to_save_ckpt, saving_file, format="ckpt")


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert to safetensors format checkpoint"""
    layer_st_map = layers_model_file_map(input_path)
    torch.set_default_dtype(torch.bfloat16)

    dtype = config["dtype"]
    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_st_map = defaultdict()
    for layer_id in range(total_num_layers):
        if layer_id == 0:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif layer_id == total_num_layers - 1:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=True)
        else:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=False)

        # first dequanting weights
        pt_layer_weights = dequant_layer_weights(layer_id, pt_layer_weights)

        ms_layer_weights = defaultdict()
        if layer_id == 0:
            ms_layer_weights.update(_model_preprocess_pt_to_ms(pt_layer_weights))
        ms_layer_weights.update(_mla_pt_to_ms(layer_id, pt_layer_weights, config))
        ms_layer_weights.update(_mlp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id > num_layers - 1:
            ms_layer_weights.update(_mtp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id == total_num_layers - 1:
            ms_layer_weights.update(_model_postprocess_pt_to_ms(pt_layer_weights))

        to_save_ckpt = []
        saving_file = f"ms-model-{layer_id+1:05d}-of-{total_num_layers:05d}.safetensors"
        for name in list(ms_layer_weights.keys()):
            value = ms_layer_weights.pop(name).to(torch.float32).numpy()
            tmp_dtype = dtype
            if "norm" in name or "router.dense" in name or "topk_bias" in name:
                tmp_dtype = ms.float32
            to_save_ckpt.append({'name': name, 'data': ms.Tensor(value, dtype=tmp_dtype)})
            converted_st_map[name] = saving_file

        ms.save_checkpoint(to_save_ckpt, os.path.join(output_path, saving_file), format="safetensors")
        print(f"saving weights in layer-{layer_id} to file {saving_file}")

    converted_model_index_file = os.path.join(output_path, f"ms-model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)


def convert_pt_to_ms(input_path, output_path, config=None):
    """convert hf weight to ms."""
    if config is None:
        config = default_config
    os.makedirs(output_path, exist_ok=True)

    save_format = config['save_format']
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    if save_format == "ckpt":
        ms_ckpt_convertor(input_path, output_path, config)

    if save_format == "safetensors":
        ms_safetensors_convertor(input_path, output_path, config)

    print("Finish converting Huggingface checkpoints into mindspore checkpoints!")


def convert_ms_to_gmm(input_path, output_path):
    """convert ms routing ffn weight for gmm."""
    params = ms.load_checkpoint(input_path)
    for k, v in params.items():
        if 'feed_forward.routed_experts.ffn.w1.weight' in k or \
                'feed_forward.routed_experts.ffn.w2.weight' in k or \
                'feed_forward.routed_experts.ffn.w3.weight' in k:
            orig_tensor = ms.Tensor(v)
            gmm_tensor = orig_tensor.transpose((0, 2, 1))
            params[k] = ms.Parameter(gmm_tensor)
            print(f"\rConvertion finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_routed_experts', default=256, type=int)
    parser.add_argument('--torch_ckpt_path', default=None, type=str)
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str)
    parser.add_argument('--use_gemm', action='store_true')
    parser.add_argument('--pre_ckpt_path', default=None, type=str)
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("--num_layers", default=61, type=int)
    parser.add_argument("--num_nextn_predict_layers", default=1, type=int)
    parser.add_argument("--first_k_dense_replace", default=3, type=int)
    parser.add_argument("--n_head", default=128, type=int)
    parser.add_argument("--qk_nope_head_dim", default=128, type=int)
    parser.add_argument("--qk_rope_head_dim", default=64, type=int)
    parser.add_argument("--v_head_dim", default=128, type=int)
    parser.add_argument("--save_format", default="safetensors", choices=["safetensors", "ckpt"])

    args = parser.parse_args()

    if args.pre_ckpt_path:
        convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
    else:
        for key in default_config:
            default_config[key] = getattr(args, key, default_config[key])
        default_config['dtype'] = dtype_map.get(default_config['dtype'], default_config['dtype'])

        convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path, config=default_config)
