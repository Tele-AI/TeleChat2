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
import os
import argparse
import mindspore as ms
import torch
from mindformers.tools.utils import str2bool
from mindformers.utils.convert_utils import pt2ms


def convert_pretrained_weight(pth_file="swin_base_patch4_window7_224.pth", ms_ckpt_path="swin_base_p4w7.ckpt",
                              dtype=None):
    """
    convert swin_base_p4w7 weights from pytorch to mindspore
    pytorch required.
    """
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    for k, v in state_dict['model'].items():
        if 'decoder.0' in k:
            k = k.replace('decoder.0', 'decoder')
        if 'patch_embed.' in k:
            k = k.replace('proj', 'projection')
        if 'relative_position' in k:
            k = k.replace('relative_position', 'relative_position_bias.relative_position')
        if 'norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if '.mlp' in k:
            if '.fc1' in k:
                k = k.replace('.fc1', '.mapping')
                if "weight" in k:
                    v = v.transpose(-1, 0)
            if '.fc2' in k:
                k = k.replace('.fc2', '.projection')
                if "weight" in k:
                    v = v.transpose(-1, 0)
        v = pt2ms(v, dtype)
        if '.qkv' not in k:
            ms_ckpt.append({'name': k, 'data': v})
        else:
            length = len(v)
            ms_ckpt.append({'name': k.replace('.qkv', '.q'), 'data': ms.Tensor(v[:length // 3])})
            ms_ckpt.append({'name': k.replace('.qkv', '.k'), 'data': ms.Tensor(v[length // 3:length // 3 * 2])})
            ms_ckpt.append({'name': k.replace('.qkv', '.v'), 'data': ms.Tensor(v[length // 3 * 2:])})
    if not os.path.exists(ms_ckpt_path):
        try:
            ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Save checkpoint to {ms_ckpt_path} failed"
                ", please check the path, permissions, and checkpoint data validity."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Save checkpoint to {ms_ckpt_path} failed"
                "with unknown error, please check the path,"
                " permissions, and checkpoint data validity."
            ) from e


def convert_finetuned_weight(pth_file="swin_base_patch4_window7_224.pth", ms_ckpt_path="swin_base_p4w7.ckpt",
                             dtype=None):
    """
    convert swin_base_p4w7 weights from pytorch to mindspore
    pytorch required.
    """
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    for k, v in state_dict['model'].items():
        if 'head' in k:
            ms_ckpt.append({'name': k, 'data': pt2ms(v, dtype)})
            continue
        if 'patch_embed.' in k:
            k = k.replace('proj', 'projection')
        if 'relative_position' in k:
            k = k.replace('relative_position', 'relative_position_bias.relative_position')
        if 'norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if '.mlp' in k:
            if '.fc1' in k:
                k = k.replace('.fc1', '.mapping')
                if "weight" in k:
                    v = v.transpose(-1, 0)
            if '.fc2' in k:
                k = k.replace('.fc2', '.projection')
                if "weight" in k:
                    v = v.transpose(-1, 0)
        v = pt2ms(v, dtype)
        if '.qkv' not in k:
            ms_ckpt.append({'name': 'encoder.' + k, 'data': v})
        else:
            length = len(v)
            ms_ckpt.append({'name': 'encoder.' + k.replace('.qkv', '.q'), 'data': ms.Tensor(v[:length // 3])})
            ms_ckpt.append(
                {'name': 'encoder.' + k.replace('.qkv', '.k'), 'data': ms.Tensor(v[length // 3:length // 3 * 2])})
            ms_ckpt.append({'name': 'encoder.' + k.replace('.qkv', '.v'), 'data': ms.Tensor(v[length // 3 * 2:])})

    if not os.path.exists(ms_ckpt_path):
        try:
            ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Save checkpoint to {ms_ckpt_path} failed"
                ", please check the path, permissions, and checkpoint data validity."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Save checkpoint to {ms_ckpt_path} failed"
                "with unknown error, please check the path,"
                " permissions, and checkpoint data validity."
            ) from e


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """
    convert torch weight to mindspore weight

    Args:
        input_path: torch weight path
        output_path: mindspore weight path
        dtype: The dtype of th converted weight.
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
    parser.add_argument("--torch_path",
                        type=str,
                        default="swin_base_patch4_window7_224.pth",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="swin_base_p4w7.ckpt",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--is_pretrain",
                        type=str2bool,
                        default=False,
                        help="whether converting pretrained weights.")
    opt = parser.parse_args()
    convert_pt_to_ms(opt.torch_path, opt.mindspore_path, is_pretrain=opt.is_pretrain)
