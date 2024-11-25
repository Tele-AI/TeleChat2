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
"""transform ckpt"""
import os
import argparse

import mindspore as ms


def get_strategy(startegy_path, rank_id=None):
    """Merge strategy if strategy path is dir

    Args:
        startegy_path (str): The path of stategy.
        rank_id (int): The rank id of device.

    Returns:
        None or strategy path
    """
    if not startegy_path or startegy_path == "None":
        return None

    if not os.path.exists(startegy_path):
        raise ValueError(f'{startegy_path} not found!')

    if os.path.isfile(startegy_path):
        return startegy_path

    if os.path.isdir(startegy_path):
        if rank_id:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy_{rank_id}.ckpt')
        else:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy.ckpt')

        if os.path.exists(merge_path):
            os.remove(merge_path)

        ms.merge_pipeline_strategys(startegy_path, merge_path)
        return merge_path

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_strategy',
                        default="",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_dir',
                        default="",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_ckpt_dir',
                        default="",
                        type=str,
                        help='path where to save dst ckpt')
    parser.add_argument('--prefix',
                        default='checkpoint_',
                        type=str,
                        help='prefix of transformed checkpoint')
    args = parser.parse_args()

    src_ckpt_strategy = get_strategy(args.src_ckpt_strategy)
    dst_ckpt_strategy = get_strategy(args.dst_ckpt_strategy)
    src_ckpt_dir = args.src_ckpt_dir
    dst_ckpt_dir = args.dst_ckpt_dir
    prefix = args.prefix

    print(f"src_ckpt_strategy: {src_ckpt_strategy}")
    print(f"dst_ckpt_strategy: {dst_ckpt_strategy}")
    print(f"src_ckpt_dir: {src_ckpt_dir}")
    print(f"dst_ckpt_dir: {dst_ckpt_dir}")
    print(f"prefix: {prefix}")

    print("......Start transform......")
    ms.transform_checkpoints(src_ckpt_dir, dst_ckpt_dir, prefix, src_ckpt_strategy, dst_ckpt_strategy)
    print("......Transform succeed!......")
