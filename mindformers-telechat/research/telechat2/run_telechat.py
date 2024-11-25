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
    yaml_path = os.path.expanduser(args.config)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    config = MindFormerConfig(os.path.realpath(yaml_path))
    if args.use_parallel is not None:
        config.use_parallel = args.use_parallel
    if args.device_id is not None:
        config.context.device_id = args.device_id
    if args.load_checkpoint is not None:
        config.load_checkpoint = args.load_checkpoint
    if args.src_strategy is not None and os.path.exists(args.src_strategy):
        config.src_strategy_path_or_dir = args.src_strategy
    if args.auto_trans_ckpt is not None:
        config.auto_trans_ckpt = args.auto_trans_ckpt
    if args.remote_save_url is None:
        config.remote_save_url = args.remote_save_url
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = 'InferenceMatmulSplit'
    # init context
    build_context(config)

    # start task
    trainer = Trainer(args=config,
                      task=args.task,
                      train_dataset=args.train_dataset)
    trainer.train(train_checkpoint=config.load_checkpoint, auto_trans_ckpt=config.auto_trans_ckpt,
                  resume_training=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='telechat2/finetune_telechat_115b.yaml', type=str,
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
    parser.add_argument('--train_dataset', default=None, type=str,
                        help='set train dataset.')
    parser.add_argument('--remote_save_url', default=None, type=str,
                        help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()

    main()
