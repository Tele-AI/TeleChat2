"""Convert internvl2 weight."""
import os
import argparse
from glob import glob
from safetensors import safe_open
import mindspore as ms
import mindspore.common.dtype as mstype
from mindformers.utils.convert_utils import pt2ms
from mindformers.tools import logger


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
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    weight_name = weight_name.replace('.norm.', '.norm_out.')
    weight_name = weight_name.replace('output.', 'lm_head.')
    weight_name = weight_name.replace('.tok_embeddings.weight', '.tok_embeddings.embedding_weight')
    if 'mlp' in weight_name:
        weight_name = weight_name.replace('mlp1.0.bias', 'mlp1.layer_norm.beta')
        weight_name = weight_name.replace('mlp1.0.weight', 'mlp1.layer_norm_gamma')
        weight_name = weight_name.replace('mlp1.1.bias', 'mlp1.adapter1.bias')
        weight_name = weight_name.replace('mlp1.1.weight', 'mlp1.adapter1.weight')
        weight_name = weight_name.replace('mlp1.3.bias', 'mlp1.adapter2.bias')
        weight_name = weight_name.replace('mlp1.3.weight', 'mlp1.adapter2.weight')
    return weight_name


def convert_pt_to_ms(input_path, output_path, dtype=mstype.float32):
    """convert hf weight to ms."""
    ckpts = glob(os.path.join(input_path, '*.safetensors'))
    ckpts.sort()
    model_params = dict()
    for ckpt in ckpts:
        with safe_open(ckpt, framework='pt', device='cpu') as f:
            for k in f.keys():
                model_params[k] = f.get_tensor(k)

    ckpt_list = []
    for name, value in model_params.items():
        name = name_replace(name)
        value = pt2ms(value, dtype)
        ckpt_list.append({'name': name, 'data': value})
    logger.info('Start convert checkpoint')
    ms.save_checkpoint(ckpt_list, output_path)
    logger.info(f"Convert finished, the output is saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="internvl2 convert script")
    parser.add_argument('--torch_ckpt_dir', default=None)
    parser.add_argument('--mindspore_ckpt_path', default='./internvl2.ckpt')
    parser.add_argument('--dtype', default='float32', type=str, choices=['float16', 'float32', 'bfloat16'])
    args = parser.parse_args()
    dtype_map = {'float16': ms.float16, 'float32': ms.float32, 'bfloat16': ms.bfloat16}
    convert_pt_to_ms(
        input_path=args.torch_ckpt_path,
        output_path=args.mindspore_ckpt_path,
        dtype=dtype_map.get(args.dtype, None)
    )
