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
"""Convert checkpoint from torch/MicroSoft"""
import collections
import os
import argparse
import mindspore as ms
import torch
from mindformers.tools.utils import str2bool
from mindformers.utils.convert_utils import ms2pt


# pylint: disable=W0613
def convert_pretrained_weight(ms_ckpt_path="swin_base_p4w7.ckpt", pth_file="swin_base_patch4_window7_224.pth",
                              dtype=None):
    """
    convert swin_base_p4w7 weights from mindspore to pytorch
    """
    pt_ckpt = {}
    ms_ckpt = ms.load_checkpoint(ms_ckpt_path)
    attention_dict = collections.defaultdict(lambda: {})
    for name, value in ms_ckpt.items():
        value = ms2pt(value, dtype)
        if '.q' in name:
            name = name.replace('.q', '.qkv')
            attention_dict[name]['q'] = value
        if '.k' in name:
            name = name.replace('.k', '.qkv')
            attention_dict[name]['k'] = value
        if '.v' in name:
            name = name.replace('.v', '.qkv')
            attention_dict[name]['v'] = value
        if name in attention_dict:
            if len(attention_dict[name]) == 3:
                value = torch.cat((attention_dict[name]['q'], attention_dict[name]['k'], attention_dict[name]['v']))
            else:
                continue

        if '.mlp' in name:
            if '.projection' in name:
                name = name.replace('.projection', '.fc2')
                if 'weight' in name:
                    value = value.transpose(-1, 0)
            if '.mapping' in name:
                name = name.replace('.mapping', '.fc1')
                if "weight" in name:
                    value = value.transpose(-1, 0)
        if 'norm' in name:
            if '.beta' in name:
                name = name.replace('.beta', '.bias')
            if '.gamma' in name:
                name = name.replace('.gamma', '.weight')
        if 'relative_position_bias.relative_position' in name:
            name = name.replace('relative_position_bias.relative_position', 'relative_position')
        if 'patch_embed.' in name:
            name = name.replace('projection', 'proj')
        if 'decoder' in name:
            name = name.replace('decoder', 'decoder.0')
        pt_ckpt[name] = value
    state_dict = {'model': pt_ckpt}
    if not os.path.exists(pth_file):
        try:
            torch.save(state_dict, pth_file)
            print(f"Save checkpoint to {pth_file}.")
        except OSError as e:
            raise RuntimeError(
                f"Save checkpoint to {pth_file} failed, please check the path and permissions."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error occurred when saving checkpoint to {pth_file}."
            ) from e


def convert_finetuned_weight(ms_ckpt_path="swin_base_p4w7.ckpt", pth_file="swin_base_patch4_window7_224.pth",
                             dtype=None):
    """
    convert swin_base_p4w7 weights from mindspore to pytorch
    """
    pt_ckpt = {}
    ms_ckpt = ms.load_checkpoint(ms_ckpt_path)
    attention_dict = collections.defaultdict(lambda: {})
    for name, value in ms_ckpt.items():
        value = ms2pt(value, dtype)
        if 'head' in name:
            pt_ckpt[name] = value
            continue
        if '.q' in name:
            name = name.replace('.q', '.qkv')
            attention_dict[name]['q'] = value
        if '.k' in name:
            name = name.replace('.k', '.qkv')
            attention_dict[name]['k'] = value
        if '.v' in name:
            name = name.replace('.v', '.qkv')
            attention_dict[name]['v'] = value
        if name in attention_dict:
            if len(attention_dict[name]) == 3:
                value = torch.cat((attention_dict[name]['q'], attention_dict[name]['k'], attention_dict[name]['v']))
            else:
                continue

        name = name.replace('encoder.', '')
        if '.mlp' in name:
            if '.projection' in name:
                name = name.replace('.projection', '.fc2')
                if 'weight' in name:
                    value = value.transpose(-1, 0)
            if '.mapping' in name:
                name = name.replace('.mapping', '.fc1')
                if "weight" in name:
                    value = value.transpose(-1, 0)
        if 'norm' in name:
            if '.beta' in name:
                name = name.replace('.beta', '.bias')
            if '.gamma' in name:
                name = name.replace('.gamma', '.weight')
        if 'relative_position_bias.relative_position' in name:
            name = name.replace('relative_position_bias.relative_position', 'relative_position')
        if 'patch_embed.' in name:
            name = name.replace('projection', 'proj')
        pt_ckpt[name] = value

    state_dict = {'model': pt_ckpt}
    if not os.path.exists(pth_file):
        try:
            torch.save(state_dict, pth_file)
            print(f"Save checkpoint to {pth_file}.")
        except OSError as e:
            raise RuntimeError(
                f"Save checkpoint to {pth_file} failed, please check the path and permissions."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error occurred when saving checkpoint to {pth_file}."
            ) from e


def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert mindspore weight to torch weight

    Args:
        input_path: mindspore weight path
        output_path: torch weight path
        dtype: The dtype of the converted weight.
        **kwargs: extra args

    Returns:
        None
    """
    is_pretrain = kwargs.pop('is_pretrain', False)
    if is_pretrain:
        convert_pretrained_weight(input_path, output_path, dtype)
    else:
        convert_finetuned_weight(input_path, output_path, dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="swin weight convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="swin_base_p4w7.ckpt",
                        help="The mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        default="swin_base_patch4_window7_224.pth",
                        required=True,
                        help="The output torch checkpoint path.")
    parser.add_argument("--is_pretrain",
                        type=str2bool,
                        default=False,
                        help="whether converting pretrained weights.")
    opt = parser.parse_args()
    convert_ms_to_pt(opt.mindspore_path, opt.torch_path, is_pretrain=opt.is_pretrain)
