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
"""Convert whisper weight from ms to pt."""
import argparse
import torch
import numpy as np
import mindspore as ms


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=ms.float32, **kwargs):
    """convert whisper torch model to mindspore ckpt"""
    torch_weight = ms.load_checkpoint(input_path)
    state_dict = {}
    for key, weight in torch_weight.items():
        array = weight.numpy()
        if "embed_tokens.embedding_weight" in key:
            key = key.replace("embedding_weight", "weight")
        if "embed_positions.embedding_weight" in key:
            key = key.replace("embedding_weight", "weight")
        if "layer_norm" in key:
            key = key.replace("gamma", "weight")
            key = key.replace("beta", "bias")
        if "conv1.weight" in key or "conv2.weight" in key:
            array = np.squeeze(array, axis=2)
        state_dict[key] = torch.from_numpy(array).cpu().to(dtype)

    torch.save(state_dict, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper convert script")
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    convert_ms_to_pt(input_path=args.input_path, output_path=args.output_path)
