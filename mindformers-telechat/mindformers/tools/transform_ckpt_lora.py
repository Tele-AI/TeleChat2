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
"""transform lora_ckpt"""
import os
import argparse
from collections import OrderedDict

import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.ops as P
from mindformers.tools.logger import logger


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


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_strategy',
                        default="",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_path_or_dir',
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
    parser.add_argument('--lora_scaling',
                        default=1,
                        type=float,
                        help='scale of lora when merge model weight, default is lora_alpha/lora_rank')
    args = parser.parse_args()

    src_ckpt_strategy = get_strategy(args.src_ckpt_strategy)
    dst_ckpt_strategy = get_strategy(args.dst_ckpt_strategy)
    src_ckpt_path_or_dir = args.src_ckpt_path_or_dir
    dst_ckpt_dir = args.dst_ckpt_dir
    prefix = args.prefix
    lora_scaling = args.lora_scaling

    logger.info(f"src_ckpt_strategy: {src_ckpt_strategy}")
    logger.info(f"dst_ckpt_strategy: {dst_ckpt_strategy}")
    logger.info(f"src_ckpt_path_or_dir: {src_ckpt_path_or_dir}")
    logger.info(f"dst_ckpt_dir: {dst_ckpt_dir}")
    logger.info(f"prefix: {prefix}")

    if not os.path.isdir(src_ckpt_path_or_dir):
        logger.info("......Only Need MergeLora......")
        src_lora_ckpt_path = src_ckpt_path_or_dir
    else:
        logger.info("......Need Merge&Trans......")
        logger.info("......Start Transckpt......")
        ms.transform_checkpoints(src_ckpt_path_or_dir, dst_ckpt_dir, prefix, src_ckpt_strategy, dst_ckpt_strategy)
        logger.info("......Complete Trans&Save......")
        src_lora_ckpt_path = os.path.join(dst_ckpt_dir, "rank_0", prefix + "0.ckpt")
        logger.info("src_lora_ckpt_path---------------", src_lora_ckpt_path)
    logger.info("......Start Merge Lorackpt......")
    param_dict = ms.load_checkpoint(src_lora_ckpt_path)
    lora_keys = [k for k in param_dict if 'lora_a' in k]
    non_lora_keys = [k for k in param_dict if 'lora_' not in k]
    param_dict_lora = OrderedDict()
    for k in non_lora_keys:
        param_dict_lora[k] = param_dict[k].clone()
    for k in lora_keys:
        if k.split('.')[0] in ['adam_m', 'adam_v']:
            continue
        logger.info(f'Merging {k}')
        original_key = k.replace('_lora_a', '').replace('mindpet_delta', 'weight')
        if original_key not in param_dict:
            raise ValueError(f"The original_key should in the param_dict, but got {original_key}.")
        lora_a_key = k
        lora_b_key = k.replace('lora_a', 'lora_b')
        original_value = param_dict_lora[original_key]
        param_dict_lora[original_key] = Parameter(Tensor(P.add(original_value, P.mm(param_dict[lora_b_key], \
                                         param_dict[lora_a_key]) * lora_scaling), original_value.dtype), \
                                         name=original_key)
    logger.info("......Start save merged ckpt......")
    save_checkpoint_file_name = os.path.join(dst_ckpt_dir, 'merged_lora.ckpt')
    ms.save_checkpoint(param_dict_lora, save_checkpoint_file_name)
    logger.info("......Merge succeed!.......")
