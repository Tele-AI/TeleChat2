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
"""Convert checkpoint from torch/facebook"""
import argparse
import torch
import mindspore as ms

from mindformers.models.vit.convert_weight import replace_process


def convert_pt_to_ms(input_path, output_path, dtype=None):
    """
    convert mae_vit_base_p16 weights from pytorch to mindspore
    pytorch and GPU required.
    """
    param_dict = torch.load(input_path, map_location=torch.device("cpu"))
    new_dict = replace_process(param_dict, dtype, 'mae')
    ms.save_checkpoint(new_dict, output_path)
    print("Weights conversion completes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=True,
                        default="mae_pretrain_vit_base.ckpt",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--torch_pth_path",
                        type=str,
                        default="mae_pretrain_vit_base.pth",
                        required=True,
                        help="The torch checkpoint path.")
    opt = parser.parse_args()

    convert_pt_to_ms(opt.torch_pth_path, opt.ms_ckpt_path)
