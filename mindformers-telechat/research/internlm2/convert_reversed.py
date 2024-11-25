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
Convert InternLM2 weight.
Support mindspore format.
"""

import argparse
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt


def _name_replace(name: str):
    """replace mindformers parameter name to huggingface."""
    name = name.replace('tok_embeddings.embedding_weight', 'tok_embeddings.weight')
    name = name.replace('w', 'wqkv')
    name = name.replace('.norm_out.', '.norm.')
    name = name.replace('lm_head', 'output')
    return name


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert mindspore weights files to huggingface."""
    model_hf = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_hf.items():
        print('Parameter (name=%s, shape=%s, dtype=%s, requires_grad=%s)' % (
            name, value.data.shape, value.data.dtype, value.data.requires_grad))
        value = ms2pt(value, dtype)

        hfname = _name_replace(name)
        if hfname != name:
            print('name:  %s->%s' % (name, hfname))
        state_dict[hfname] = value

    print('Saving converted weights to %s...' % output_path)
    torch.save(state_dict, output_path)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindspore_ckpt_path",
                        default="./internlm2_7b.ckpt",
                        help="The ms checkpoint path.")
    parser.add_argument("--torch_ckpt_path",
                        required=True,
                        help="The output checkpoint path.")

    args = parser.parse_args()
    convert_ms_to_pt(args.mindspore_ckpt_path, args.torch_ckpt_path)
