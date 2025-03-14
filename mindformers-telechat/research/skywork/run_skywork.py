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
"""Skywork-13B Train/Finetune/Eval/Predict/Export scripts."""

import argparse

from mindformers import Trainer, MindFormerConfig
from mindformers.tools.utils import check_in_modelarts, set_remote_save_url, str2bool
from mindformers.core.context import build_context


def main(task='text_generation',
         config='run_skywork_13b.yaml',
         run_mode='predict',
         pet_method='',
         use_parallel=False,
         resume=False,
         auto_trans_ckpt=False,
         train_dataset="",
         ckpt=None,
         eval_dataset='',
         predict_data='',
         max_length=512,
         op=True,
         remote_save_url=None,
         device_id=None,
         use_past=False,
         batch_size=None):
    """main function."""
    # 适配aicc
    if check_in_modelarts() and remote_save_url:
        print("remote_save_url is %s, the output file will be uploaded to here.", remote_save_url)
        set_remote_save_url(remote_save_url)

    config_args = MindFormerConfig(config)

    if ckpt:
        config_args.load_checkpoint = ckpt
    if device_id:
        config_args.context.device_id = device_id
    config_args.parallel.enable_parallel_optimizer = op
    config_args.auto_trans_ckpt = auto_trans_ckpt
    config_args.use_parallel = use_parallel
    config_args.model.model_config.checkpoint_name_or_path = config_args.load_checkpoint
    config_args.model.model_config.use_past = use_past
    if batch_size:
        config_args.model.model_config.batch_size = batch_size

    # 环境初始化
    build_context(config_args)

    # 定义任务，预先准备好相应数据集
    if run_mode == 'train':
        trainer = Trainer(args=config_args,
                          task=task,
                          train_dataset=train_dataset,
                          pet_method=pet_method)
        trainer.train(train_checkpoint=config_args.load_checkpoint, auto_trans_ckpt=config_args.auto_trans_ckpt,
                      resume_training=resume)
    elif run_mode == 'finetune':
        trainer = Trainer(args=config_args,
                          task=task,
                          train_dataset=train_dataset,
                          pet_method=pet_method)
        print(trainer)
        trainer.finetune(finetune_checkpoint=config_args.load_checkpoint, auto_trans_ckpt=config_args.auto_trans_ckpt,
                         resume_training=resume)
    elif run_mode == 'eval':
        trainer = Trainer(args=config_args,
                          task=task,
                          eval_dataset=eval_dataset)
        trainer.evaluate(eval_checkpoint=config_args.load_checkpoint)
    elif run_mode == 'predict':
        trainer = Trainer(args=config_args,
                          task=task)
        result = trainer.predict(predict_checkpoint=config_args.load_checkpoint, input_data=predict_data,
                                 max_length=int(max_length))
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='run_skywork_13b.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='predict', type=str,
                        help='set run mode for model.')
    parser.add_argument('--pet_method', default='', type=str,
                        help='set pet method for low parameter finetune.')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=False, type=str2bool,
                        help='whether auto trans ckpt.')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--eval_dataset', default='', type=str,
                        help='set eval dataset.')
    parser.add_argument('--predict_data', default='', type=str,
                        help='input predict data.')
    parser.add_argument('--device_id', default=None, type=int,
                        help='set device id.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='batch_size for export mindir.')
    parser.add_argument('--optimizer_parallel', default=True, type=str2bool,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--remote_save_url', default="", type=str,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--use_past', default=False, type=str2bool,
                        help='whether use past. Default: False')
    args = parser.parse_args()
    print(args)

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         pet_method=args.pet_method,
         use_parallel=args.use_parallel,
         resume=args.resume,
         auto_trans_ckpt=args.auto_trans_ckpt,
         train_dataset=args.train_dataset,
         ckpt=args.load_checkpoint,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         max_length=args.predict_length,
         op=args.optimizer_parallel,
         remote_save_url=args.remote_save_url,
         device_id=args.device_id,
         use_past=args.use_past,
         batch_size=args.batch_size)
