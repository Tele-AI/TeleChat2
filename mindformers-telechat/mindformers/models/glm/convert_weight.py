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
"""Convert checkpoint from torch/huggingface"""
import argparse
import os

from transformers import AutoModel
import mindspore as ms

from mindformers.utils.convert_utils import pt2ms


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """ Convert pytorch model file to MindSpore model file. """
    model_hf = AutoModel.from_pretrained(os.path.dirname(input_path), trust_remote_code=True).half()

    print('parameter convert....')
    ms_param = []
    for k, v in model_hf.state_dict().items():
        v = pt2ms(v, dtype)
        if "word_embeddings.weight" in k:
            k = k.replace("weight", "embedding_table")
        if "post_attention_layernorm" in k or "input_layernorm" in k or "final_layernorm" in k:
            k = k.replace("weight", "gamma")
            k = k.replace("bias", "beta")
        if "input_layernorm" in k:
            v = v.half()
        print(f"{k} {v.dtype}")
        ms_param.append({"name": k, "data": v})

    print('saving ms ckpt....')
    ms.save_checkpoint(ms_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChatGLM6B weight convert script")
    parser.add_argument("--pt_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The torch checkpoint path.")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=False,
                        default="ms_glm_6b.ckpt",
                        help='The output mindspore checkpoint path.')

    opt = parser.parse_args()
    convert_pt_to_ms(opt.pt_ckpt_path, opt.ms_ckpt_path)
