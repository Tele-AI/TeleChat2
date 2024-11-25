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

from mindformers.models.vit.convert_reversed import convert_ms_to_pt as convert_func


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert mae_vit_base_p16 weights from mindspore to pytorch.
    """
    convert_func(input_path, output_path, dtype=dtype, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--torch_pth_path",
                        type=str,
                        default="mae_pretrain_vit_base.pth",
                        required=True,
                        help="The output torch checkpoint path.")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=True,
                        default="mae_pretrain_vit_base.ckpt",
                        help="The mindspore checkpoint path.")
    opt = parser.parse_args()

    convert_ms_to_pt(opt.ms_ckpt_path, opt.torch_pth_path)
