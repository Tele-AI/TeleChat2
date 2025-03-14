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
transform huggingface model to mindspore ckpt.
"""
import os
import argparse
import mindspore as ms
import torch
from transformers import AutoModelForCausalLM


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.q_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
    weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
    weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
    weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
    weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
    weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
    weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
    weight_name = weight_name.replace('mlp.experts.', 'feed_forward.ffn.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
    weight_name = weight_name.replace('mlp.gate.', 'feed_forward.routed_experts.router.dense.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


def convert_pt_to_ms(input_path, output_path, num_routed_experts=160, dtype=ms.bfloat16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        model_hf = AutoModelForCausalLM.from_pretrained(os.path.dirname(input_path), device_map="cpu",
                                                        torch_dtype=torch.bfloat16, attn_implementation="eager")
        model_hf = model_hf.to('cpu')
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False

    ckpt_list = []

    count = 0
    all_num = 3 * int(num_routed_experts)
    list_w1 = []
    list_w2 = []
    list_w3 = []

    for name, value in model_hf.named_parameters():
        print("origin_name:", name, value.shape)
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'

        if 'feed_forward.ffn' in name: # concat expert

            if "gate_proj" in name: # gate_proj
                list_w1.append(value.detach())
            if "down_proj" in name: # down_proj
                list_w2.append(value.detach())
            if "up_proj" in name: # up_proj
                list_w3.append(value.detach())
            count = count + 1
            if count == all_num:
                str_front = name.split('ffn')[0]
                print(str_front)
                name_w1 = str_front + 'routed_experts.ffn.w1.weight' # gate_proj
                name_w2 = str_front + 'routed_experts.ffn.w2.weight' # up_proj
                name_w3 = str_front + 'routed_experts.ffn.w3.weight' # down_proj
                w1_value = torch.stack(list_w1, 0).to(torch.float32).cpu().numpy()
                print(f'\rprocessing parameter: {name_w1} {w1_value.shape}     ')
                ckpt_list.append({'name': name_w1, 'data': ms.Tensor(w1_value, dtype=dtype)})
                w2_value = torch.stack(list_w2, 0).to(torch.float32).cpu().numpy()
                print(f'\rprocessing parameter: {name_w2} {w2_value.shape}     ')
                ckpt_list.append({'name': name_w2, 'data': ms.Tensor(w2_value, dtype=dtype)})
                w3_value = torch.stack(list_w3, 0).to(torch.float32).cpu().numpy()
                print(f'\rprocessing parameter: {name_w3} {w3_value.shape}     ')
                ckpt_list.append({'name': name_w3, 'data': ms.Tensor(w3_value, dtype=dtype)})
                count = 0
                list_w1 = []
                list_w2 = []
                list_w3 = []
        elif "norm" in name or "dense" in name:
            ms_value = value.to(torch.float32).detach().cpu().numpy()
            ckpt_list.append({'name': name, 'data': ms.Tensor(ms_value, dtype=ms.float32)})
        else:
            ms_value = value.to(torch.float32).detach().cpu().numpy()
            ckpt_list.append({'name': name, 'data': ms.Tensor(ms_value, dtype=ms.bfloat16)})

        print("converted_name: ", name, value.shape)

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished,"
          "the mindspore checkpoint is saved in '{output_path}'.", flush=True)

    return True


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


def mla_weight_split(input_path, output_path, n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
    """split qkv weight in mla module."""
    print("Begin MLA weight process.")
    import copy
    params = ms.load_checkpoint(input_path)
    params_bk = copy.deepcopy(params)
    for name, value in params_bk.items():
        # l2q_proj process
        if ".attention.l2q_proj." in name:
            value_tensor = ms.Tensor(value)
            q_head = qk_nope_head_dim + qk_rope_head_dim
            value_tensor = value_tensor.reshape(n_head, q_head, -1)
            value_nope, value_pe = value_tensor[:, :qk_nope_head_dim, :], value_tensor[:, qk_nope_head_dim:, :]
            value_pe = value_pe.reshape(value_pe.shape[0], value_pe.shape[1] // 2, 2, -1).transpose(0, 2, 1, 3)
            value_nope = value_nope.reshape(-1, value_nope.shape[-1])
            value_pe = value_pe.reshape(-1, value_pe.shape[-1])
            name_lt = name.split(".attention.l2q_proj.")
            params[name_lt[0] + ".attention.l2q_nope_proj." + name_lt[-1]] = ms.Parameter(value_nope)
            params[name_lt[0] + ".attention.l2q_pe_proj." + name_lt[-1]] = ms.Parameter(value_pe)
            del params[name]

        # .attention.kv2l. process
        if ".attention.kv2l." in name:
            value_tensor = ms.Tensor(value)
            kv_lora_rank = value_tensor.shape[0] - qk_rope_head_dim
            value_latent_kv, value_k_pe = value_tensor[:kv_lora_rank, :], value_tensor[kv_lora_rank:, :]
            value_k_pe = value_k_pe.reshape(value_k_pe.shape[0] // 2, 2, -1).transpose(1, 0, 2)
            value_k_pe = value_k_pe.reshape(-1, value_k_pe.shape[-1])
            name_lt = name.split(".attention.kv2l.")
            params[name_lt[0] + ".attention.kv2l_k_pe." + name_lt[-1]] = ms.Parameter(value_k_pe)
            params[name_lt[0] + ".attention.kv2l_latent_kv." + name_lt[-1]] = ms.Parameter(value_latent_kv)
            del params[name]

        # .attention.lkv2kv. process
        if ".attention.lkv2kv." in name:
            value_tensor = ms.Tensor(value)
            lkv2kv_head = qk_nope_head_dim + v_head_dim
            value_tensor = value_tensor.reshape(n_head, lkv2kv_head, -1)
            value_k_nope, value_v = value_tensor[:, :qk_nope_head_dim, :], value_tensor[:, qk_nope_head_dim:, :]
            value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
            value_v = value_v.reshape(-1, value_v.shape[-1])
            name_lt = name.split(".attention.lkv2kv.")
            params[name_lt[0] + ".attention.lkv2kv_k_nope." + name_lt[-1]] = ms.Parameter(value_k_nope)
            params[name_lt[0] + ".attention.lkv2kv_v." + name_lt[-1]] = ms.Parameter(value_v)
            del params[name]
    ms.save_checkpoint(params, output_path)
    print(f"\rMLA weight trans finished, the mindspore ckpt is save in '{output_path}'.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_routed_experts', default=160)
    parser.add_argument('--torch_ckpt_path', default=None)
    parser.add_argument('--mindspore_ckpt_path', default=None)
    parser.add_argument('--use_gmm', action='store_true')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("--n_head", default=128)
    parser.add_argument("--qk_nope_head_dim", default=128)
    parser.add_argument("--qk_rope_head_dim", default=64)
    parser.add_argument("--v_head_dim", default=128)
    parser.add_argument("mla_weight_split_flag", default=False)
    args = parser.parse_args()
    if args.mla_weight_split_flag:
        if os.path.isdir(args.pre_ckpt_path):
            rank_ids = os.listdir(args.pre_ckpt_path)
            for rank_id in rank_ids:
                id_num = rank_id.split('_')[1]
                input_ckpt = os.path.join(args.pre_ckpt_path, rank_id + f'/checkpoint_{id_num}.ckpt')
                if not os.path.exists(input_ckpt):
                    print(f"[WARNING]: ckpt file {input_ckpt} does not exist, skip rank {rank_id}.")
                    continue
                output_dir = os.path.join(args.mindspore_ckpt_path, rank_id)
                output_ckpt = os.path.join(output_dir, f'checkpoint_{id_num}.ckpt')
                os.makedirs(output_dir, exist_ok=True)
                mla_weight_split(input_path=input_ckpt,
                                 output_path=output_ckpt,
                                 n_head=args.n_head,
                                 qk_nope_head_dim=args.qk_nope_head_dim,
                                 qk_rope_head_dim=args.qk_rope_head_dim,
                                 v_head_dim=args.v_head_dim)
        else:
            mla_weight_split(input_path=args.pre_ckpt_path,
                             output_path=args.mindspore_ckpt_path,
                             n_head=args.n_head,
                             qk_nope_head_dim=args.qk_nope_head_dim,
                             qk_rope_head_dim=args.qk_rope_head_dim,
                             v_head_dim=args.v_head_dim)

    if args.pre_ckpt_path:
        if os.path.isdir(args.pre_ckpt_path):
            rank_ids = os.listdir(args.pre_ckpt_path)
            for rank_id in rank_ids:
                id_num = rank_id.split('_')[1]
                input_ckpt = os.path.join(args.pre_ckpt_path, rank_id + f'/checkpoint_{id_num}.ckpt')
                if not os.path.exists(input_ckpt):
                    continue
                output_dir = os.path.join(args.mindspore_ckpt_path, rank_id)
                output_ckpt = os.path.join(output_dir, f'checkpoint_{id_num}.ckpt')
                os.makedirs(output_dir, exist_ok=True)
                convert_ms_to_gmm(input_ckpt, output_ckpt)
        else:
            convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
    else:
        convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path,
                         num_routed_experts=args.num_routed_experts)
