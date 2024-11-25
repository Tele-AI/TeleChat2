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
"""Convert PanguAlpha checkpoint from official"""
import os
import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer

from mindformers import AutoConfig, AutoModel


def convert_pretrained_weight(config_path_or_name,
                              strategy_file,
                              checkpoint_dir,
                              npy_dir,
                              ckpt_save_path="pangualpha.ckpt"):
    """
    convert pangu alpha weight for mindformers
    """

    # organize ckpt directory
    for i in range(512):
        rank_dir = os.path.join(checkpoint_dir, f'rank_{i}')
        ckpt_path = os.path.join(checkpoint_dir, f'filerted_{i}.ckpt')
        new_ckpt_path = os.path.join(rank_dir, f'filerted_{i}.ckpt')
        os.mkdir(rank_dir)
        os.rename(ckpt_path, new_ckpt_path)

    temp_path = os.path.join(checkpoint_dir, 'temp')
    os.mkdir(temp_path)

    ms.transform_checkpoints(checkpoint_dir, temp_path,
                             'merged_pangu', strategy_file)

    checkpoint_name_or_path = os.path.join(
        temp_path, 'rank_0/merged_pangu0.ckpt')

    model_config = AutoConfig.from_pretrained(config_path_or_name)
    model_config.checkpoint_name_or_path = checkpoint_name_or_path
    pangu_model = AutoModel.from_config(model_config)

    load_embedding_from_ckpt(pangu_model, npy_dir)

    ms.save_checkpoint(pangu_model, ckpt_save_path)


def load_embedding_from_ckpt(model, npy_dir):
    r"""load the weights from the checkpoint"""
    def load_param(path):
        if os.path.exists(path):
            p_table = np.load(path)
            table_param = Tensor(p_table, mstype.float32)
        else:
            raise ValueError(f"{path} file not exits, "
                             f"please check whether embedding file exit.")
        return table_param

    # three embedding needed to be loaded
    # Loading the embedding table from the ckpt path:
    position_embedding_path = os.path.join(npy_dir, 'position_embedding.npy')
    word_embedding_path = os.path.join(npy_dir, 'word_embedding.npy')
    top_query_embedding_path = os.path.join(npy_dir, 'top_query_embedding.npy')
    model.backbone.embedding.word_embedding.embedding_table = Parameter(
        initializer(load_param(word_embedding_path),
                    [model.config.vocab_size, model.config.hidden_size]),
        name='word_embedding_table',
        parallel_optimizer=False)
    print("load word_embedding_table succeed.")

    model.backbone.embedding.position_embedding.embedding_table = Parameter(
        initializer(load_param(position_embedding_path),
                    [model.config.seq_length, model.config.hidden_size]),
        name='position_embedding_table',
        parallel_optimizer=False)
    print("load position_embedding_table succeed.")

    model.backbone.top_query_embedding.embedding_table = Parameter(
        initializer(load_param(top_query_embedding_path),
                    [model.config.seq_length, model.config.hidden_size]),
        name='query_embedding_table',
        parallel_optimizer=False)
    print("load query_embedding_table succeed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="swin weight convert script")
    parser.add_argument("--config_path_or_name",
                        type=str,
                        default="",
                        required=True,
                        help="config name or path. Eg: 'pangualpha_13b' or 'path/to/config'.")
    parser.add_argument("--official_strategy_path",
                        type=str,
                        default="",
                        required=True,
                        help="The official strategy file path.")
    parser.add_argument("--official_ckpt_dir",
                        type=str,
                        default="",
                        required=True,
                        help="The directory  where saved official ckpt files.")
    parser.add_argument("--official_npy_dir",
                        type=str,
                        default="",
                        required=True,
                        help="The folder where saved 3 embedding_table.py.")
    parser.add_argument("--ckpt_save_path",
                        type=str,
                        default="pangualpha.ckpt",
                        help="The output mindspore checkpoint path.")
    args = parser.parse_args()

    convert_pretrained_weight(args.config_path_or_name,
                              args.official_strategy_path,
                              args.official_ckpt_dir,
                              args.official_npy_dir,
                              args.ckpt_save_path)
