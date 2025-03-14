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
Support huggingface format and Meta format.
"""

import json
import argparse
import mindspore as ms
from mindformers.utils.convert_utils import pt2ms

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')

    name = name.replace('.cross_attn_mlp_gate', '.cross_attn_ff_gate')
    name = name.replace('.cross_attn.q_proj.', '.attention.wq.')
    name = name.replace('.cross_attn.k_proj.', '.attention.wk.')
    name = name.replace('.cross_attn.v_proj.', '.attention.wv.')
    name = name.replace('.cross_attn.o_proj.', '.attention.wo.')
    name = name.replace('.cross_attn.q_norm.', '.attention.wq_norm.')
    name = name.replace('.cross_attn.k_norm.', '.attention.wk_norm.')
    return name


def name_replace_vision(name: str):
    """replace hf param name to ms."""
    name = name.replace('input_layernorm.weight', 'input_layernorm.gamma')
    name = name.replace('input_layernorm.bias', 'input_layernorm.beta')
    name = name.replace('layernorm_post.weight', 'layernorm_post.gamma')
    name = name.replace('layernorm_post.bias', 'layernorm_post.beta')
    name = name.replace('post_attention_layernorm.weight', 'post_attention_layernorm.gamma')
    name = name.replace('post_attention_layernorm.bias', 'post_attention_layernorm.beta')
    name = name.replace('layernorm_pre.weight', 'layernorm_pre.gamma')
    name = name.replace('layernorm_pre.bias', 'layernorm_pre.beta')
    name = name.replace('tile_embedding.weight', 'tile_embedding.embedding_table')
    name = name.replace('.embedding.weight', '.embedding.embedding_table')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import MllamaForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "Failed to load Hugging Face checkpoint. "
            "Please make sure the 'transformers' library is installed and available."
        ) from e
    except Exception as e:
        raise RuntimeError("Unexpected error occurred when loading Hugging Face checkpoint.") from e
    try:
        model_hf = MllamaForConditionalGeneration.from_pretrained(input_path)
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{input_path}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        if "language_model" in name:
            name = name_replace(name)
        elif "vision_model" in name:
            name = name_replace_vision(name)

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_path', default='./llama_model/llama-13b-hf/hf.bin')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--dtype', default='fp32')
    args = parser.parse_args()

    convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path,
                     dtype=dtype_map.get(args.dtype))
