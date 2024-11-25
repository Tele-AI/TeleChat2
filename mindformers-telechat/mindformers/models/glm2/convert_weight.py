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
import mindspore as ms
from tqdm import tqdm
from transformers import AutoModel

from mindformers.utils.convert_utils import pt2ms


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """ Convert pytorch model file to MindSpore model file. """
    input_dir = os.path.dirname(input_path)
    model = AutoModel.from_pretrained(input_dir, trust_remote_code=True)

    print('parameter convert....')
    ms_param = []
    for k, v in tqdm(model.state_dict().items()):
        if "word_embeddings.weight" in k:
            k = k.replace("word_embeddings.weight", "embedding_weight")
        ms_param.append({"name": k, "data": pt2ms(v, dtype=dtype)})

    ms.save_checkpoint(ms_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLM2/3 weight convert script")
    parser.add_argument("--torch_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help='The output mindspore checkpoint path.')

    opt = parser.parse_args()
    convert_pt_to_ms(opt.torch_ckpt_path, opt.mindspore_ckpt_path)
