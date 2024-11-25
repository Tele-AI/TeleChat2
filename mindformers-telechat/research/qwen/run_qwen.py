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
"""Qwen task script"""
import argparse
import os
import shutil

from mindformers import Trainer, MindFormerConfig
from mindformers.core.context import build_context
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import check_in_modelarts, str2bool, set_remote_save_url
from mindformers.tools.cloud_adapter import cloud_monitor

# pylint: disable=W0611
import qwen_model
# pylint: disable=W0611
import qwen_tokenizer
# pylint: disable=W0611
import qwen_config
# pylint: disable=W0611
import optim


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        import moxing as mox

        obs_strategy_dir = os.path.join(config.remote_save_url, "strategy")
        if mox.file.exists(obs_strategy_dir) and config.local_rank == 0:
            mox.file.remove(obs_strategy_dir, recursive=True)
        obs_transformed_ckpt_dir = os.path.join(config.remote_save_url, "transformed_checkpoint")
        if mox.file.exists(obs_transformed_ckpt_dir) and config.local_rank == 0:
            mox.file.remove(obs_transformed_ckpt_dir, recursive=True)
        mox.file.make_dirs(obs_strategy_dir)
        mox.file.make_dirs(obs_transformed_ckpt_dir)
    else:
        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(strategy_dir)
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(transformed_ckpt_dir)
        os.makedirs(strategy_dir, exist_ok=True)
        os.makedirs(transformed_ckpt_dir, exist_ok=True)


def expand_input_list(input_list, batch_size):
    """Expand 'input_list' to a list of size 'batch_size'."""
    if len(input_list) < batch_size:
        repeat_time = batch_size // len(input_list) + 1
        input_list = input_list * repeat_time
    input_list = input_list[:batch_size]
    return input_list


def run_predict(task, input_list, batch_size, ckpt, max_length):
    prompt = expand_input_list(input_list, batch_size)

    result = task.predict(input_data=prompt,
                          predict_checkpoint=ckpt,
                          max_length=int(max_length))
    print(result)


# pylint: disable=C0326
@cloud_monitor()
def main(task='text_generation',
         config='run_qwen_7b.yaml',
         run_mode='predict',
         use_parallel=False,
         use_past=None,
         ckpt=None,
         auto_trans_ckpt=None,
         remote_save_url=None,
         vocab_file=None,
         predict_data='',
         seq_length=None,
         batch_size=None,
         max_length=512,
         train_dataset='',
         device_id=0,
         do_sample=None,
         top_k=None,
         top_p=None):
    """main function."""

    yaml_path = os.path.expanduser(config)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    config = MindFormerConfig(os.path.realpath(yaml_path))
    if vocab_file:
        config.processor.tokenizer.vocab_file = vocab_file
    vocab_file = config.processor.tokenizer.vocab_file
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(vocab_file)

    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id

    # init context
    build_context(config)

    if check_in_modelarts() and remote_save_url:
        print("remote_save_url is '%s', the output file will be uploaded to here.", remote_save_url)
        set_remote_save_url(remote_save_url)
        config.remote_save_url = remote_save_url

    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
        if config.auto_trans_ckpt:
            clear_auto_trans_output(config)

    if use_past is not None:
        config.model.model_config.use_past = use_past
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if batch_size is not None:
        config.model.model_config.batch_size = batch_size

    if do_sample is not None:
        config.model.model_config.do_sample = do_sample
    if top_k is not None:
        config.model.model_config.top_k = top_k
    if top_p is not None:
        config.model.model_config.top_p = top_p

    if run_mode in ['train', 'finetune']:
        config.model.model_config.use_past = False

    if run_mode == 'predict':
        task = Trainer(args=config, task=task)

        batch_size = config.model.model_config.batch_size
        if isinstance(predict_data, list):
            for i in range(0, len(predict_data), batch_size):
                input_list = predict_data[i:i + batch_size]
                run_predict(task, input_list, batch_size, ckpt, max_length)
        else:
            if predict_data:
                run_predict(task, [predict_data, ], batch_size, ckpt, max_length)
            else:
                while True:
                    user_input = input("Please enter your predict data:\n> ")
                    run_predict(task, [user_input, ], batch_size, ckpt, max_length)
    elif run_mode == 'finetune':
        if batch_size is not None:
            config.runner_config.batch_size = batch_size

        trainer = Trainer(args=config, task=task, train_dataset=train_dataset)
        trainer.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=auto_trans_ckpt)
    else:
        raise NotImplementedError(f"run_mode '${run_mode}' not supported yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='run_qwen_7b.yaml', type=str,
                        help='config file path.')
    parser.add_argument('--run_mode', default='predict', type=str,
                        help='set run mode for model.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--remote_save_url', default="", type=str,
                        help='OBS url to store/exchange transformed checkpoint files')
    parser.add_argument('--vocab_file', default="", type=str,
                        help='tokenizer model')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq_length')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='batch_size')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--device_id', default=-1, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')

    predict_group = parser.add_argument_group(title="Predict options")
    predict_group.add_argument('--predict_data', default='', type=str, nargs='+',
                               help='input predict data.')
    predict_group.add_argument('--predict_length', default=512, type=int,
                               help='max length for predict output.')
    predict_group.add_argument('--use_past', default=None, type=str2bool,
                               help='use_past')
    predict_group.add_argument('--do_sample', default=None, type=str2bool,
                               help='do_sample')
    predict_group.add_argument('--top_k', default=None, type=int,
                               help='top_k')
    predict_group.add_argument('--top_p', default=None, type=float,
                               help='top_p')

    train_group = parser.add_argument_group(title="Train/finetune options")
    train_group.add_argument('--train_dataset', default='', type=str,
                             help='set train dataset.')
    train_group.add_argument('--optimizer_parallel', default=False, type=str2bool,
                             help='whether use optimizer parallel. Default: False')

    args = parser.parse_args()

    if args.device_id == -1:
        args.device_id = int(os.getenv("DEVICE_ID", "0"))

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         use_parallel=args.use_parallel,
         use_past=args.use_past,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         remote_save_url=args.remote_save_url,
         vocab_file=args.vocab_file,
         predict_data=args.predict_data,
         seq_length=args.seq_length,
         batch_size=args.batch_size,
         max_length=args.predict_length,
         device_id=args.device_id,
         train_dataset=args.train_dataset,
         do_sample=args.do_sample,
         top_k=args.top_k,
         top_p=args.top_p)
