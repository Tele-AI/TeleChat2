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
import collections

import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert mae_vit_base_p16 weights from mindspore to pytorch
    """
    param_dict = ms.load_checkpoint(input_path)

    state_dict = {}
    attention_dict = collections.defaultdict(lambda: {})
    for name, value in param_dict.items():
        value = ms2pt(value, dtype)
        if 'attention' in name:
            name = name.replace('attention', 'attn')
            if 'projection' in name:
                name = name.replace('projection', 'proj')
                if 'weight' in name:
                    value = value.transpose(-1, 0)
        if "output" in name:
            name = name.replace("output", "mlp")
            if "mapping" in name:
                name = name.replace("mapping", "fc1")
                if "weight" in name:
                    value = value.transpose(-1, 0)
            elif "projection" in name:
                name = name.replace("projection", "fc2")
                if "weight" in name:
                    value = value.transpose(-1, 0)
        if 'beta' in name:
            name = name.replace('beta', 'bias')
        elif 'gamma' in name:
            name = name.replace('gamma', 'weight')
        if 'layernorm' in name:
            name = name.replace('layernorm', 'norm')
        if "head" not in name:
            name = name.lstrip('vit.')
        if name in ("cls_tokens", "mask_tokens"):
            name = name[:-1]

        if '.dense1' in name:
            name = name.replace('.dense1', '.qkv')
            attention_dict[name]['dense1'] = value
            continue
        if '.dense2' in name:
            name = name.replace('.dense2', '.qkv')
            attention_dict[name]['dense2'] = value
            continue
        if '.dense3' in name:
            name = name.replace('.dense3', '.qkv')
            attention_dict[name]['dense3'] = value
            continue
        state_dict[name] = value
    for name, value_dict in attention_dict.items():
        state_dict[name] = torch.cat((value_dict['dense1'], value_dict['dense2'], value_dict['dense3']))

    pth_dict = collections.OrderedDict()
    pth_dict['model'] = state_dict

    torch.save(pth_dict, output_path)
    print("Weights conversion completes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=True,
                        default="mae_pretrain_vit_base.ckpt",
                        help="The mindspore checkpoint path.")
    parser.add_argument("--torch_pth_path",
                        type=str,
                        default="mae_pretrain_vit_base.pth",
                        required=True,
                        help="The output torch checkpoint path.")
    opt = parser.parse_args()

    convert_ms_to_pt(opt.ms_ckpt_path, opt.torch_pth_path)
