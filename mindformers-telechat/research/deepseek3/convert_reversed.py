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
transform mindspore ckpt to huggingface model.
"""
import argparse
import json
import os
from collections import defaultdict
from glob import glob
import warnings

import mindspore as ms
from mindspore.ops.operations import Cast
import torch
from safetensors.torch import save_file

ms.set_context(device_target='CPU')
cpu_cast = Cast().set_device('CPU')

dtype_map = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16
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
    'dtype': torch.bfloat16,
    'use_gemm': False,
    'load_format': "safetensors"
}


def plain_name_replace(weight_name: str):
    """Weight name replacing for pre/post-process module"""
    weight_name = weight_name.replace('tok_embeddings.embedding_weight', 'embed_tokens.weight')
    weight_name = weight_name.replace('model.norm_out.weight', 'model.norm.weight')
    return weight_name


def mla_name_replace(weight_name: str):
    """Weight name replacing for MLA module weights"""
    weight_name = weight_name.replace('.attention.q_proj.', '.self_attn.q_proj.')
    weight_name = weight_name.replace('.attention.q2l_proj.', '.self_attn.q_a_proj.')
    weight_name = weight_name.replace('.attention.lq_norm.', '.self_attn.q_a_layernorm.')
    weight_name = weight_name.replace('.attention.l2q_proj.', '.self_attn.q_b_proj.')
    weight_name = weight_name.replace('.attention.kv2l.', '.self_attn.kv_a_proj_with_mqa.')
    weight_name = weight_name.replace('.attention.lkv_norm.', '.self_attn.kv_a_layernorm.')
    weight_name = weight_name.replace('.attention.lkv2kv.', '.self_attn.kv_b_proj.')
    weight_name = weight_name.replace('.attention.wo.', '.self_attn.o_proj.')
    weight_name = weight_name.replace('.attention_norm.', '.input_layernorm.')
    weight_name = weight_name.replace('.ffn_norm.', '.post_attention_layernorm.')
    return weight_name


def mlp_name_replace(weight_name: str):
    """Weight name replacing for MLP module, including MoE"""
    weight_name = weight_name.replace('feed_forward.w1.', 'mlp.gate_proj.')
    weight_name = weight_name.replace('feed_forward.w2.', 'mlp.down_proj.')
    weight_name = weight_name.replace('feed_forward.w3.', 'mlp.up_proj.')
    weight_name = weight_name.replace('feed_forward.shared_experts.w1.', 'mlp.shared_experts.gate_proj.')
    weight_name = weight_name.replace('feed_forward.shared_experts.w2.', 'mlp.shared_experts.down_proj.')
    weight_name = weight_name.replace('feed_forward.shared_experts.w3.', 'mlp.shared_experts.up_proj.')
    weight_name = weight_name.replace('feed_forward.routed_experts.router.dense.weight', 'mlp.gate.weight')
    weight_name = weight_name.replace('feed_forward.routed_experts.router.router.topk_bias',
                                      'mlp.gate.e_score_correction_bias')
    return weight_name


def mtp_name_replace(weight_name: str, current_layer_id: int, mtp_layer_id: int):
    """replace weight name for MultiPredictionToken module"""
    weight_name = weight_name.replace(f"model.mtp_hidden_fusers.{mtp_layer_id}.norm_emb",
                                      f"model.layers.{current_layer_id}.enorm")
    weight_name = weight_name.replace(f"model.mtp_hidden_fusers.{mtp_layer_id}.norm",
                                      f"model.layers.{current_layer_id}.hnorm")
    weight_name = weight_name.replace(f"model.mtp_hidden_fusers.{mtp_layer_id}.dense",
                                      f"model.layers.{current_layer_id}.eh_proj")
    return weight_name


def load_data_ms(file_name):
    return ms.load_checkpoint(file_name, format="safetensors")


def layers_model_file_map(file_path, config):
    """Get weight-file map"""
    num_layers = config["num_layers"]
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "param_name_map.json")
    if not os.path.exists(weight_map_file):
        weight_map_file = os.path.join(file_path, "ms-model.safetensors.index.json")

    if os.path.exists(weight_map_file):
        with open(weight_map_file) as f:
            weights_map = json.load(f)
        try:
            weights_map = weights_map["weight_map"]
        except KeyError:
            pass
    else:
        warnings.warn(f"Cannot find weight map file eighther param_name_map.json or " \
                    f"ms-model.safetensors.index.json in path {file_path}, " \
                    f"Trying to load one safetensor file ...")
        files = sorted(glob(os.path.join(file_path, "*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors files found in path {file_path}")

        weight_file = files[0].split("/")[-1]
        keys = load_data_ms(os.path.join(file_path, weight_file)).keys()
        weights_map = {}
        for k in keys:
            weights_map[k] = weight_file

    for weight_key, value in weights_map.items():
        if weight_key.startswith("model.layers."):
            layer_name = int(weight_key.split('model.layers.')[1].split('.')[0])
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        elif weight_key.startswith("model.mtp_hidden_fusers."):
            mtp_layer_name = int(weight_key.split('model.mtp_hidden_fusers.')[1].split('.')[0])
            layer_name = num_layers + mtp_layer_name
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def read_matched_file(layer_st_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_first:
        st_file_list.extend(list(layer_st_map["model.tok_embeddings.embedding_weight"]))
    if is_last:
        st_file_list.extend(list(layer_st_map["model.norm_out.weight"]))
        st_file_list.extend(list(layer_st_map["lm_head.weight"]))
    st_file_list = list(set(st_file_list))
    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_ms(st_file)
        weights.update(current_weight)
    return weights


def _mla_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MLA module"""
    n_head = config['n_head']
    qk_nope_head_dim = config['qk_nope_head_dim']
    qk_rope_head_dim = config['qk_rope_head_dim']
    v_head_dim = config['v_head_dim']
    dtype = config['dtype']

    qk_nope_key = f"model.layers.{layer_id}.attention.l2q_nope_proj.weight"
    qk_rope_key = f"model.layers.{layer_id}.attention.l2q_pe_proj.weight"
    latent_kv_key = f"model.layers.{layer_id}.attention.kv2l_latent_kv.weight"
    k_rope_key = f"model.layers.{layer_id}.attention.kv2l_k_pe.weight"
    k_nope_key = f"model.layers.{layer_id}.attention.lkv2kv_k_nope.weight"
    v_key = f"model.layers.{layer_id}.attention.lkv2kv_v.weight"

    q_a_proj_key = f"model.layers.{layer_id}.attention.q2l_proj.weight"
    kv_a_proj_key = f"model.layers.{layer_id}.attention.kv2l.weight"
    o_proj_key = f"model.layers.{layer_id}.attention.wo.weight"
    q_a_layernorm_key = f"model.layers.{layer_id}.attention.lq_norm.weight"
    kv_a_layernorm_key = f"model.layers.{layer_id}.attention.lkv_norm.weight"
    q_b_proj_key = f"model.layers.{layer_id}.attention.l2q_proj.weight"
    kv_b_proj_key = f"model.layers.{layer_id}.attention.lkv2kv.weight"
    input_norm_key = f"model.layers.{layer_id}.attention_norm.weight"
    post_attn_norm_key = f"model.layers.{layer_id}.ffn_norm.weight"

    qk_nope = cpu_cast(ms_layer_weights.pop(qk_nope_key), ms.float32).numpy()
    qk_rope = cpu_cast(ms_layer_weights.pop(qk_rope_key), ms.float32).numpy()
    latent_kv = cpu_cast(ms_layer_weights.pop(latent_kv_key), ms.float32).numpy()
    k_rope = cpu_cast(ms_layer_weights.pop(k_rope_key), ms.float32).numpy()
    k_nope = cpu_cast(ms_layer_weights.pop(k_nope_key), ms.float32).numpy()
    v = cpu_cast(ms_layer_weights.pop(v_key), ms.float32).numpy()

    q_a_proj = cpu_cast(ms_layer_weights.pop(q_a_proj_key), ms.float32).numpy()
    o_proj = cpu_cast(ms_layer_weights.pop(o_proj_key), ms.float32).numpy()
    q_a_layernorm = cpu_cast(ms_layer_weights.pop(q_a_layernorm_key), ms.float32).numpy()
    kv_a_layernorm = cpu_cast(ms_layer_weights.pop(kv_a_layernorm_key), ms.float32).numpy()
    input_norm = cpu_cast(ms_layer_weights.pop(input_norm_key), ms.float32).numpy()
    post_attn_norm = cpu_cast(ms_layer_weights.pop(post_attn_norm_key), ms.float32).numpy()


    mla_weight_dict = defaultdict()
    # merge qk_nope, qk_rope into q_b_proj
    qk_rope = torch.from_numpy(qk_rope).to(dtype).reshape(n_head, 2, qk_rope_head_dim // 2, -1)
    qk_rope = qk_rope.permute(0, 2, 1, 3).reshape(n_head, qk_rope_head_dim, -1)
    qk_nope = torch.from_numpy(qk_nope).to(dtype).reshape(n_head, qk_nope_head_dim, -1)
    q_b_proj = torch.cat([qk_nope, qk_rope], dim=1).reshape(-1, qk_nope.shape[-1])
    q_b_proj_key = mla_name_replace(q_b_proj_key)
    mla_weight_dict[q_b_proj_key] = q_b_proj.clone()

    # merge latent_kv, k_rope into kv_a_proj
    k_rope = torch.from_numpy(k_rope).to(dtype).reshape(2, k_rope.shape[0] // 2, -1).permute(1, 0, 2)
    k_rope = k_rope.reshape(-1, k_rope.shape[-1])
    latent_kv = torch.from_numpy(latent_kv).to(dtype)
    kv_a_proj = torch.cat([latent_kv, k_rope], dim=0)
    kv_a_proj_key = mla_name_replace(kv_a_proj_key)
    mla_weight_dict[kv_a_proj_key] = kv_a_proj.clone()

    # merge k_nope, v into kv_b_proj
    k_nope = torch.from_numpy(k_nope).to(dtype).reshape(n_head, qk_nope_head_dim, -1)
    v = torch.from_numpy(v).to(dtype).reshape(n_head, v_head_dim, -1)
    kv_b_proj = torch.cat([k_nope, v], dim=1).reshape(-1, k_nope.shape[-1])
    kv_b_proj_key = mla_name_replace(kv_b_proj_key)
    mla_weight_dict[kv_b_proj_key] = kv_b_proj.clone()

    # process q_a_proj, o_proj, and layernorms
    q_a_proj_key = mla_name_replace(q_a_proj_key)
    mla_weight_dict[q_a_proj_key] = torch.from_numpy(q_a_proj).to(dtype).clone()
    o_proj_key = mla_name_replace(o_proj_key)
    mla_weight_dict[o_proj_key] = torch.from_numpy(o_proj).to(dtype).clone()
    q_a_layernorm_key = mla_name_replace(q_a_layernorm_key)
    mla_weight_dict[q_a_layernorm_key] = torch.from_numpy(q_a_layernorm).to(dtype).clone()
    kv_a_layernorm_key = mla_name_replace(kv_a_layernorm_key)
    mla_weight_dict[kv_a_layernorm_key] = torch.from_numpy(kv_a_layernorm).to(dtype).clone()
    input_norm_key = mla_name_replace(input_norm_key)
    mla_weight_dict[input_norm_key] = torch.from_numpy(input_norm).to(dtype).clone()
    post_attn_norm_key = mla_name_replace(post_attn_norm_key)
    mla_weight_dict[post_attn_norm_key] = torch.from_numpy(post_attn_norm).to(dtype).clone()

    return mla_weight_dict


def _mlp_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MLP/MoE module"""
    num_routed_experts = config['num_routed_experts']
    first_k_dense_replace = config['first_k_dense_replace']
    dtype = config['dtype']

    mlp_weight_dict = defaultdict()
    if layer_id < first_k_dense_replace:
        gate_proj_key = f"model.layers.{layer_id}.feed_forward.w1.weight"
        up_proj_key = f"model.layers.{layer_id}.feed_forward.w3.weight"
        down_proj_key = f"model.layers.{layer_id}.feed_forward.w2.weight"
        gate_proj = cpu_cast(ms_layer_weights.pop(gate_proj_key), ms.float32).numpy()
        up_proj = cpu_cast(ms_layer_weights.pop(up_proj_key), ms.float32).numpy()
        down_proj = cpu_cast(ms_layer_weights.pop(down_proj_key), ms.float32).numpy()

        gate_proj_key = mlp_name_replace(gate_proj_key)
        up_proj_key = mlp_name_replace(up_proj_key)
        down_proj_key = mlp_name_replace(down_proj_key)
        mlp_weight_dict[gate_proj_key] = torch.from_numpy(gate_proj).to(dtype).clone()
        mlp_weight_dict[up_proj_key] = torch.from_numpy(up_proj).to(dtype).clone()
        mlp_weight_dict[down_proj_key] = torch.from_numpy(down_proj).to(dtype).clone()
    else:
        router_weight_key = f"model.layers.{layer_id}.feed_forward.routed_experts.router.dense.weight"
        router_correct_bias_key = f"model.layers.{layer_id}.feed_forward.routed_experts.router.router.topk_bias"
        shared_experts_gate_proj_key = f"model.layers.{layer_id}.feed_forward.shared_experts.w1.weight"
        shared_experts_up_proj_key = f"model.layers.{layer_id}.feed_forward.shared_experts.w3.weight"
        shared_experts_down_proj_key = f"model.layers.{layer_id}.feed_forward.shared_experts.w2.weight"
        router_weight = cpu_cast(ms_layer_weights.pop(router_weight_key), ms.float32).numpy()
        router_weight = router_weight[:num_routed_experts, :]
        router_correct_bias = cpu_cast(ms_layer_weights.pop(router_correct_bias_key), ms.float32).numpy()
        router_correct_bias = router_correct_bias[:num_routed_experts]
        shared_experts_gate_proj = cpu_cast(ms_layer_weights.pop(shared_experts_gate_proj_key), ms.float32).numpy()
        shared_experts_up_proj = cpu_cast(ms_layer_weights.pop(shared_experts_up_proj_key), ms.float32).numpy()
        shared_experts_down_proj = cpu_cast(ms_layer_weights.pop(shared_experts_down_proj_key), ms.float32).numpy()

        # replace name and store
        router_weight_key = mlp_name_replace(router_weight_key)
        router_correct_bias_key = mlp_name_replace(router_correct_bias_key)
        shared_experts_gate_proj_key = mlp_name_replace(shared_experts_gate_proj_key)
        shared_experts_up_proj_key = mlp_name_replace(shared_experts_up_proj_key)
        shared_experts_down_proj_key = mlp_name_replace(shared_experts_down_proj_key)
        mlp_weight_dict[router_weight_key] = torch.from_numpy(router_weight).to(dtype).clone()
        mlp_weight_dict[router_correct_bias_key] = torch.from_numpy(router_correct_bias).to(dtype).clone()
        mlp_weight_dict[shared_experts_gate_proj_key] = torch.from_numpy(shared_experts_gate_proj).to(dtype).clone()
        mlp_weight_dict[shared_experts_up_proj_key] = torch.from_numpy(shared_experts_up_proj).to(dtype).clone()
        mlp_weight_dict[shared_experts_down_proj_key] = torch.from_numpy(shared_experts_down_proj).to(dtype).clone()

        # routed experts
        expert_gate_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        expert_up_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"
        expert_down_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        expert_gate_proj = cpu_cast(ms_layer_weights.pop(expert_gate_proj_key), ms.float32).numpy()
        expert_up_proj = cpu_cast(ms_layer_weights.pop(expert_up_proj_key), ms.float32).numpy()
        expert_down_proj = cpu_cast(ms_layer_weights.pop(expert_down_proj_key), ms.float32).numpy()
        expert_gate_proj = torch.from_numpy(expert_gate_proj).to(dtype).reshape(num_routed_experts,
                                                                                -1, expert_gate_proj.shape[-1])
        expert_up_proj = torch.from_numpy(expert_up_proj).to(dtype).reshape(num_routed_experts,
                                                                            -1, expert_up_proj.shape[-1])
        expert_down_proj = torch.from_numpy(expert_down_proj).to(dtype).reshape(num_routed_experts,
                                                                                -1, expert_down_proj.shape[-1])

        for expert_id in range(num_routed_experts):
            gate_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
            up_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
            down_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
            mlp_weight_dict[gate_proj_key] = expert_gate_proj[expert_id, ...].clone()
            mlp_weight_dict[up_proj_key] = expert_up_proj[expert_id, ...].clone()
            mlp_weight_dict[down_proj_key] = expert_down_proj[expert_id, ...].clone()

    return mlp_weight_dict


def _mtp_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MTP module"""
    num_layers = config["num_layers"]
    dtype = config['dtype']

    mtp_layer_id = layer_id - num_layers
    # ignore the shared emb_weights and lm head in mtp layers
    enorm_key = f"model.mtp_hidden_fusers.{mtp_layer_id}.norm_emb.weight"
    hnorm_key = f"model.mtp_hidden_fusers.{mtp_layer_id}.norm.weight"
    e_proj_key = f"model.mtp_hidden_fusers.{mtp_layer_id}.dense.weight"

    enorm = cpu_cast(ms_layer_weights.pop(enorm_key), ms.float32).numpy()
    hnorm = cpu_cast(ms_layer_weights.pop(hnorm_key), ms.float32).numpy()
    e_proj = cpu_cast(ms_layer_weights.pop(e_proj_key), ms.float32).numpy()

    mtp_weight_dict = defaultdict()
    enorm_key = mtp_name_replace(enorm_key, layer_id, mtp_layer_id)
    hnorm_key = mtp_name_replace(hnorm_key, layer_id, mtp_layer_id)
    e_proj_key = mtp_name_replace(e_proj_key, layer_id, mtp_layer_id)
    mtp_weight_dict[enorm_key] = torch.from_numpy(enorm).to(dtype).clone()
    mtp_weight_dict[hnorm_key] = torch.from_numpy(hnorm).to(dtype).clone()
    mtp_weight_dict[e_proj_key] = torch.from_numpy(e_proj).to(dtype).clone()

    emb_weight_key = "model.tok_embeddings.embedding_weight"
    final_norm_key = "model.norm_out.weight"
    lm_head_key = "lm_head.weight"
    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    final_norm = cpu_cast(ms_layer_weights.get(final_norm_key), ms.float32).numpy()
    lm_head = cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()

    shared_embed_key = f"model.layers.{layer_id}.embed_tokens.weight"
    shared_head_norm_key = f"model.layers.{layer_id}.shared_head.norm.weight"
    shared_head_key = f"model.layers.{layer_id}.shared_head.head.weight"
    mtp_weight_dict[shared_embed_key] = torch.from_numpy(emb_weight).to(dtype).clone()
    mtp_weight_dict[shared_head_key] = torch.from_numpy(lm_head).to(dtype).clone()
    mtp_weight_dict[shared_head_norm_key] = torch.from_numpy(final_norm).to(dtype).clone()

    return mtp_weight_dict


def _model_preprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in prepross module"""
    dtype = config['dtype']
    emb_weight_key = "model.tok_embeddings.embedding_weight"
    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    emb_weight_key = plain_name_replace(emb_weight_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = torch.from_numpy(emb_weight).to(dtype).clone()

    return plain_weight_dict


def _model_postprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in postpross module"""
    dtype = config['dtype']
    final_norm_key = "model.norm_out.weight"
    lm_head_key = "lm_head.weight"
    final_norm = cpu_cast(ms_layer_weights.get(final_norm_key), ms.float32).numpy()
    lm_head = cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()

    final_norm_key = plain_name_replace(final_norm_key)
    lm_head_key = plain_name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[final_norm_key] = torch.from_numpy(final_norm).to(dtype).clone()
    plain_weight_dict[lm_head_key] = torch.from_numpy(lm_head).to(dtype).clone()

    return plain_weight_dict


def get_torch_storage_size(tensor):
    """Get tensor's storage size, requires torch >= 2.1"""
    return tensor.untyped_storage().nbytes()


def ms_ckpt_convertor(input_path, output_path, config):
    """Convert ckpt format checkpoint"""
    # for .ckpt format checkpoints, only single file is valid
    if os.path.isdir(input_path):
        raise ValueError(f"File in `.ckpt` format is valid to convert checkpoints, but get a directory!")
    ms_weights = ms.load_checkpoint(input_path, format='ckpt')

    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_st_map = defaultdict()
    converted_st_map["weight_map"] = defaultdict()
    converted_st_map["metadata"] = defaultdict()

    total_size = 0
    for layer_id in range(total_num_layers):
        pt_layer_weights = defaultdict()
        if layer_id == 0:
            pt_layer_weights.update(_model_preprocess_ms_to_pt(ms_weights, config))
        pt_layer_weights.update(_mla_ms_to_pt(layer_id, ms_weights, config))
        pt_layer_weights.update(_mlp_ms_to_pt(layer_id, ms_weights, config))
        if layer_id > num_layers - 1:
            pt_layer_weights.update(_mtp_ms_to_pt(layer_id, ms_weights, config))
        if layer_id == total_num_layers - 1:
            pt_layer_weights.update(_model_postprocess_ms_to_pt(ms_weights, config))

        saving_file_name = f"model-{layer_id+1:05d}-of-{total_num_layers:05d}.safetensors"
        for name in list(pt_layer_weights.keys()):
            converted_st_map["weight_map"][name] = saving_file_name
            total_size += get_torch_storage_size(pt_layer_weights.get(name))
        save_file(pt_layer_weights, saving_file_name)

    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, f"model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert safetensors format checkpoint"""
    # try to get weight-file map
    layer_st_map = layers_model_file_map(input_path, config)

    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_st_map = defaultdict()
    converted_st_map["weight_map"] = defaultdict()
    converted_st_map["metadata"] = defaultdict()

    total_size = 0
    for layer_id in range(total_num_layers):
        if layer_id == 0:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif 0 < layer_id < num_layers:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=False, is_last=False)
        else:
            # for mtp layers, embed weight and lm_head weight are needed for shared weights
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=True, is_last=True)
        pt_layer_weights = defaultdict()
        if layer_id == 0:
            pt_layer_weights.update(_model_preprocess_ms_to_pt(ms_layer_weights, config))
        pt_layer_weights.update(_mla_ms_to_pt(layer_id, ms_layer_weights, config))
        pt_layer_weights.update(_mlp_ms_to_pt(layer_id, ms_layer_weights, config))
        if layer_id > num_layers - 1:
            pt_layer_weights.update(_mtp_ms_to_pt(layer_id, ms_layer_weights, config))
        if layer_id == total_num_layers - 1:
            pt_layer_weights.update(_model_postprocess_ms_to_pt(ms_layer_weights, config))

        saving_file_name = f"model-{layer_id+1:05d}-of-{total_num_layers:05d}.safetensors"
        for name in list(pt_layer_weights.keys()):
            converted_st_map["weight_map"][name] = saving_file_name
            total_size += get_torch_storage_size(pt_layer_weights.get(name))
        save_file(pt_layer_weights, os.path.join(output_path, saving_file_name))
        print(f"saving weights in layer-{layer_id} to file {saving_file_name}")

    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, f"model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)


def convert_ms_to_pt(input_path, output_path, config=None):
    """convert ms weight to huggingface."""
    if config is None:
        config = default_config
    os.makedirs(output_path, exist_ok=True)

    load_format = config['load_format']
    print(f"Loading mindspore checkpoint in '{input_path}' ...")

    if load_format == "ckpt":
        ms_ckpt_convertor(input_path, output_path, config)

    if load_format == "safetensors":
        ms_safetensors_convertor(input_path, output_path, config)

    print("Finish converting mindspore checkpoints into Huggingface checkpoints!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_routed_experts', default=256, type=int)
    parser.add_argument('--torch_ckpt_path', default=None, type=str)
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str)
    parser.add_argument('--use_gemm', action='store_true')
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("--num_layers", default=61, type=int)
    parser.add_argument("--num_nextn_predict_layers", default=1, type=int)
    parser.add_argument("--first_k_dense_replace", default=3, type=int)
    parser.add_argument("--n_head", default=128, type=int)
    parser.add_argument("--qk_nope_head_dim", default=128, type=int)
    parser.add_argument("--qk_rope_head_dim", default=64, type=int)
    parser.add_argument("--v_head_dim", default=128, type=int)
    parser.add_argument("--load_format", default="safetensors", choices=["safetensors", "ckpt"])

    args = parser.parse_args()

    for key in default_config:
        default_config[key] = getattr(args, key, default_config[key])
    default_config['dtype'] = dtype_map.get(default_config['dtype'], default_config['dtype'])

    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_ckpt_path, config=default_config)
