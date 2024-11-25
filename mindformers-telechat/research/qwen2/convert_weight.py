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


dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('lm_head.', 'output.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
    weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
    weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace(
        '.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(
        f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError as e:
        raise ImportError(
            "Failed to load HuggingFace checkpoint. "
            "Please make sure the 'transformers' library is installed and available."
        ) from e

    try:
        model_hf = Qwen2ForCausalLM.from_pretrained(
            os.path.dirname(input_path))
    # pylint: disable=W0703
    except Exception as e:
        print(
            f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'
        value = value.detach().numpy()
        print(name, value.shape)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--dtype', default='bf16')
    args = parser.parse_args()
    ms_dtype = dtype_map.get(args.dtype)

    convert_pt_to_ms(input_path=args.torch_ckpt_dir,
                     output_path=args.mindspore_ckpt_path, dtype=ms_dtype)
