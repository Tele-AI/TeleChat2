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
"""Convert checkpoint from mindspore"""
import argparse

import torch as pt
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt, is_lora_param


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """ Convert MindSpore model file to pytorch mpdel file. """
    ckpt_dict = ms.load_checkpoint(input_path)

    print('parameter convert....')
    pt_param = {}
    for k, v in ckpt_dict.items():
        v = ms2pt(v, dtype)
        if "post_attention_layernorm" in k or "input_layernorm" in k or "final_layernorm" in k:
            k = k.replace("gamma", "weight")
            k = k.replace("beta", "bias")
        if "word_embeddings.embedding_table" in k:
            k = k.replace("embedding_table", "weight")
        if is_lora_param(k):
            k = k.replace('tk_delta_lora_a', 'lora_A.weight')
            k = k.replace('tk_delta_lora_b', 'lora_B.weight')
        pt_param[k] = v

    print('saving pt ckpt....')
    pt.save(pt_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChatGLM6B weight convert script")

    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=False,
                        default="ms_glm_6b.ckpt",
                        help='The input mindspore checkpoint path.')

    parser.add_argument("--pt_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The output torch checkpoint path.")

    opt = parser.parse_args()
    convert_ms_to_pt(opt.ms_ckpt_path, opt.pt_ckpt_path)
