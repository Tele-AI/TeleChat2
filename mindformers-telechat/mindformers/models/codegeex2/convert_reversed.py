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
Convert codegeex2 weight.
Support mindspore format.
"""

import argparse

import mindspore as ms
import torch

from mindformers.utils.convert_utils import ms2pt


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""

    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        print(f'\rprocessing parameter: {name}', end='', flush=True)
        if "embedding_table" in name:
            name = name.replace("embedding_table", "word_embeddings.weight")
        state_dict[name] = ms2pt(value, dtype)
    torch.save(state_dict, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', default='codegeex2.ckpt')
    parser.add_argument('--torch_bin_path', default='codegeex2.bin')
    args = parser.parse_args()

    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_bin_path)
