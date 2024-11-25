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
Support mindspore format and Meta format.
"""

import json
import argparse
import torch
import mindspore as ms


def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    if value.dtype == torch.bfloat16:
        np_value = value.to(torch.float32).numpy()
    else:
        np_value = value.detach().numpy()

    if dtype:
        return ms.Tensor(np_value, dtype=dtype)
    return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)


def ms2pt(value: ms.Tensor, dtype) -> torch.Tensor:
    """
    convert ms.Tensor to torch.Tensor with specified dtype
    """
    if value.dtype == ms.bfloat16:
        np_value = value.data.astype(ms.float32).asnumpy()
    else:
        np_value = value.data.asnumpy()

    if dtype:
        return torch.from_numpy(np_value).to(dtype)
    return torch.from_numpy(np_value).to(torch.bfloat16) if value.dtype == ms.bfloat16 else torch.from_numpy(np_value)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace ms param name to hf."""
    name = name.replace('tok_embeddings.embedding_weight', 'embed_tokens.weight')
    name = name.replace('.attention.wq.', '.self_attn.q_proj.')
    name = name.replace('.attention.wk.', '.self_attn.k_proj.')
    name = name.replace('.attention.wv.', '.self_attn.v_proj.')
    name = name.replace('.attention.wo.', '.self_attn.o_proj.')
    name = name.replace('.feed_forward.w1.', '.mlp.gate_proj.')
    name = name.replace('.feed_forward.w2.', '.mlp.down_proj.')
    name = name.replace('.feed_forward.w3.', '.mlp.up_proj.')
    name = name.replace('.attention_norm.', '.input_layernorm.')
    name = name.replace('.ffn_norm.', '.post_attention_layernorm.')
    name = name.replace('.norm_out.', '.norm.')
    return name


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{input_path}'.", flush=True)
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        name = name_replace(name)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        state_dict[name] = ms2pt(value, dtype)

    torch.save(state_dict, output_path)
    print(f"\rConvert mindspore checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--torch_ckpt_path', default='./qwen2/qwen2-hf/')
    args = parser.parse_args()
    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_ckpt_path)
