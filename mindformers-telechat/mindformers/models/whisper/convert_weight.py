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
"""Convert whisper weight from pt to ms."""
import argparse
import torch
import numpy as np
import mindspore as ms


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=ms.float32, **kwargs):
    """convert whisper torch model to mindspore ckpt"""
    weight = torch.load(input_path)
    ckpt_list = []
    for key, weight in weight.items():
        array = weight.numpy()
        if "embed_tokens.weight" in key:
            key = key.replace("weight", "embedding_weight")
        if "embed_positions.weight" in key:
            key = key.replace("weight", "embedding_weight")
        if "layer_norm" in key:
            key = key.replace("weight", "gamma")
            key = key.replace("bias", "beta")
        if "conv1.weight" in key or "conv2.weight" in key:
            array = np.expand_dims(array, axis=2)
        ckpt_list.append({'name': key, 'data': ms.Tensor(array, dtype=dtype)})
        print(f"{key} : {array.shape}")

    ms.save_checkpoint(ckpt_list, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper convert script")
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--dtype', default='float16', type=str, choices=['float16', 'float32', 'bfloat16'])
    args = parser.parse_args()

    dtype_map = {'float16': ms.float16, 'float32': ms.float32, 'bfloat16': ms.bfloat16}

    convert_pt_to_ms(input_path=args.input_path, output_path=args.output_path, dtype=dtype_map.get(args.dtype))
