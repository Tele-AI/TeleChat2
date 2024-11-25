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
Support huggingface format.
"""

import os
import argparse
import mindspore as ms

from transformers import AutoModel
from mindformers.utils.convert_utils import pt2ms


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    model_dir = os.path.dirname(input_path)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    ckpt_list = []
    for name, value in model.state_dict().items():
        print(f'\rprocessing parameter: {name}', end='', flush=True)
        if "word_embeddings.weight" in name:
            name = name.replace("word_embeddings.weight", "embedding_table")
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})
    ms.save_checkpoint(ckpt_list, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_bin_path', default='codegeex2.bin')
    parser.add_argument('--mindspore_ckpt_path', default='codegeex2.ckpt')
    args = parser.parse_args()

    convert_pt_to_ms(input_path=args.torch_bin_path, output_path=args.mindspore_ckpt_path)
