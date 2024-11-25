# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Convert checkpoint from torch/facebook"""
import argparse

import mindspore as ms
import torch


def convert_weight_sam(pth_file="sam_vit_b_01ec64.pth", ckpt_file=None):
    """
    convert sam weights from pytorch to mindspore.
    """
    ckpt = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'module' in ckpt:
        state_dict = ckpt['module']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    if not ckpt_file:
        ms_ckpt_path = pth_file.split('.')[0] + '.ckpt'
    else:
        ms_ckpt_path = ckpt_file
    ms_ckpt = []
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('norm.'):
            new_k = new_k.replace('norm.', 'fc_norm.')
        if 'norm' in new_k:
            if 'weight' in new_k:
                new_k = new_k.replace('weight', 'gamma')
            if 'bias' in k:
                new_k = new_k.replace('bias', 'beta')
        if 'prompt_encoder' in new_k:
            if 'embed' in new_k and 'weight' in new_k:
                new_k = new_k.replace('weight', 'embedding_table')
        if 'mask_decoder' in new_k:
            if 'iou_token' in new_k or 'mask_tokens' in new_k:
                new_k = new_k.replace('weight', 'embedding_table')

        ms_ckpt.append({'name': new_k, 'data': ms.Tensor(v.numpy())})

    ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
    print("Convert finished!")


def show_info(file_path="sam_vit_b_01ec64.pth"):
    """
    show info of pth or ckpt
    """
    if '.ckpt' in file_path:
        state_dict = ms.load_checkpoint(file_path)
    elif '.pth' in file_path:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
        print(info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="path/to/sam_vit_b_01ec64.pth.pth",
                        help="The torch checkpoint path.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default="path/to/sam_vit_b_01ec64.pth.ckpt",
                        help="The torch checkpoint path.")
    opt = parser.parse_args()

    convert_weight_sam(opt.torch_path, opt.ckpt_path)
