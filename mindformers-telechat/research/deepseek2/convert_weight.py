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


def convert_pt_to_ms(input_path, output_path, dtype=ms.bfloat16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        model_hf = AutoModelForCausalLM.from_pretrained(os.path.dirname(input_path),
                                                        trust_remote_code=True, device_map="cpu",
                                                        torch_dtype=torch.bfloat16, attn_implementation="eager")
        model_hf = model_hf.to('cpu')
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False

    ckpt_list = []

    count = 0
    all_num = 480
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

        print("covented_name: ", name, value.shape)

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished,"
          "the mindspore checkpoint is saved in '{output_path}'.", flush=True)

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_path', default=None)
    parser.add_argument('--mindspore_ckpt_path', default=None)
    args = parser.parse_args()
    convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path)
