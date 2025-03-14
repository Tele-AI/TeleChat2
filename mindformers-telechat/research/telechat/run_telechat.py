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
"""Telechat Train/Finetune scripts."""
import os
import sys
import argparse

# pylint: disable=W0611
from mindformers import Trainer, MindFormerConfig
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.core.context import build_context
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
MindFormerRegister.register_cls(TelechatConfig, MindFormerModuleType.CONFIG)
MindFormerRegister.register_cls(TelechatForCausalLM, MindFormerModuleType.MODELS)

sys.path.insert(0, os.getcwd().split('research')[0])


@cloud_monitor()
def main():
    """main function."""
    if not (os.path.exists(args.config) or not args.config.endswith(('.yaml', '.yml'))):
        raise ValueError("The config should exist and endswith .yaml or .yml")

    # init config
    config = MindFormerConfig(os.path.realpath(args.config))
    if args.use_parallel is not None:
        config.use_parallel = args.use_parallel
    if args.device_id is not None:
        config.context.device_id = args.device_id
    if args.load_checkpoint is not None:
        config.load_checkpoint = args.load_checkpoint
    if args.src_strategy is not None:
        config.src_strategy_path_or_dir = args.src_strategy
    if args.auto_trans_ckpt is not None:
        config.auto_trans_ckpt = args.auto_trans_ckpt
    if args.remote_save_url is not None:
        config.remote_save_url = args.remote_save_url

    # init context
    build_context(config)

    # start task
    trainer = Trainer(args=config,
                      task=task,
                      train_dataset=train_dataset)
    trainer.train(train_checkpoint=config.load_checkpoint, \
        auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='finetune_telechat_7b.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id set when run on single card. Default: 0')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--src_strategy', default=None, type=str,
                        help='strategy of load_checkpoint')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--resume', default=None, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default=None,
                        help='set train dataset.')
    parser.add_argument('--remote_save_url', default=None, type=str,
                        help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()

    main()
