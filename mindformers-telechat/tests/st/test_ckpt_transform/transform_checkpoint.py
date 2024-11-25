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
"""test transform ckpt"""
import argparse

from mindformers.tools.ckpt_transform import TransformCkpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_checkpoint',
                        default="",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_checkpoint_dir',
                        default="",
                        type=str,
                        help='path where to save dst ckpt')
    parser.add_argument('--src_strategy',
                        default=None,
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_strategy',
                        default=None,
                        help='path of dst ckpt strategy')
    parser.add_argument('--prefix',
                        default='checkpoint_',
                        type=str,
                        help='prefix of transformed checkpoint')
    parser.add_argument('--rank_id',
                        default=0,
                        type=int,
                        help='rank id')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='world size')
    parser.add_argument('--transform_process_num',
                        default=1,
                        type=int,
                        help='transform process num')
    args = parser.parse_args()

    transform_ckpt = TransformCkpt(
        rank_id=args.rank_id,
        world_size=args.world_size,
        transform_process_num=args.transform_process_num
    )

    transform_ckpt(
        src_checkpoint=args.src_checkpoint,
        dst_checkpoint_dir=args.dst_checkpoint_dir,
        src_strategy=args.src_strategy,
        dst_strategy=args.dst_strategy,
        prefix=args.prefix
    )

    print("......Transform finished!......")

main()
