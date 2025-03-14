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

"""Qwen task script"""

import argparse
import os
import shutil

from qwen2_5_tokenizer import Qwen2Tokenizer
from mindformers import Trainer, MindFormerConfig
from mindformers.core.context import build_context
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import check_in_modelarts, str2bool, set_remote_save_url

if check_in_modelarts():
    import moxing as mox


def check_file_exist(file_path):
    if not os.path.exists(file_path):
        raise ValueError("The " + file_path + " should exist.")


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        obs_strategy_dir = os.path.join(config.remote_save_url, "strategy")
        if mox.file.exists(obs_strategy_dir) and config.local_rank == 0:
            mox.file.remove(obs_strategy_dir, recursive=True)
        obs_transformed_ckpt_dir = os.path.join(
            config.remote_save_url, "transformed_checkpoint")
        if mox.file.exists(obs_transformed_ckpt_dir) and config.local_rank == 0:
            mox.file.remove(obs_transformed_ckpt_dir, recursive=True)
        mox.file.make_dirs(obs_strategy_dir)
        mox.file.make_dirs(obs_transformed_ckpt_dir)
    else:
        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(strategy_dir)
        transformed_ckpt_dir = os.path.join(
            get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(transformed_ckpt_dir)
        os.makedirs(strategy_dir, exist_ok=True)
        os.makedirs(transformed_ckpt_dir, exist_ok=True)


def main(**kwargs):
    """main function."""

    task = kwargs.get('task', 'text_generation')
    config = kwargs.get('config', 'predict_qwen2_72b_instruct.yaml')
    run_mode = kwargs.get('run_mode', 'predict')
    use_parallel = kwargs.get('use_parallel')
    use_past = kwargs.get('use_past')
    ckpt = kwargs.get('ckpt')
    auto_trans_ckpt = kwargs.get('auto_trans_ckpt')
    vocab_file = kwargs.get('vocab_file')
    merges_file = kwargs.get('merges_file')
    predict_data = kwargs.get('predict_data', '')
    seq_length = kwargs.get('seq_length')
    max_length = kwargs.get('max_length', 8192)
    train_dataset = kwargs.get('train_dataset', '')
    device_id = kwargs.get('device_id', 0)
    do_sample = kwargs.get('do_sample')
    top_k = kwargs.get('top_k')
    top_p = kwargs.get('top_p')
    remote_save_url = kwargs.get('remote_save_url')
    batch_size = kwargs.get('batch_size', 1)
    yaml_path = os.path.expanduser(config)
    check_file_exist(yaml_path)
    config = MindFormerConfig(os.path.realpath(yaml_path))
    parse_args_a(config, vocab_file, merges_file)
    # init context
    build_context(config)
    parse_args_b(config, remote_save_url, auto_trans_ckpt)
    load_config_a(config, ckpt, seq_length, use_parallel, device_id)
    load_config_b(config, use_past, do_sample, top_k, top_p)
    config.model.model_config.batch_size = batch_size
    pre_train(config, run_mode, train_dataset)

    if run_mode == 'predict':
        tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)
        task = Trainer(args=config, task=task)
        batch_input = [
            [predict_data for i in range(batch_size)],
        ]
        for prompt in batch_input:
            final_prompts = []
            for input_prompt in prompt:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt}
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                final_prompts.append(input_text)
            task.predict(input_data=final_prompts,
                         predict_checkpoint=ckpt, max_length=int(max_length), seq_length=max_length)
    elif run_mode in ['train', 'finetune']:
        trainer = Trainer(args=config)
        trainer.train()
    else:
        raise NotImplementedError(f"run_mode '${run_mode}' not supported yet.")


def load_config_a(config, ckpt, seq_length, use_parallel, device_id):
    """load config a"""
    if ckpt is not None:
        config.load_checkpoint = ckpt
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id


def load_config_b(config, use_past, do_sample, top_k, top_p):
    """load config b"""
    if use_past is not None:
        config.model.model_config.use_past = use_past
    if do_sample is not None:
        config.model.model_config.do_sample = do_sample
    if top_k is not None:
        config.model.model_config.top_k = top_k
    if top_p is not None:
        config.model.model_config.top_p = top_p


def parse_args_a(config, vocab_file, merges_file):
    """parse args a"""
    if vocab_file:
        check_file_exist(vocab_file)
        config.processor.tokenizer.vocab_file = vocab_file
    if merges_file:
        check_file_exist(merges_file)
        config.processor.tokenizer.merges_file = merges_file


def parse_args_b(config, remote_save_url, auto_trans_ckpt):
    """parse args b"""
    if check_in_modelarts() and remote_save_url:
        set_remote_save_url(remote_save_url)
        config.remote_save_url = remote_save_url

    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
        if config.auto_trans_ckpt:
            clear_auto_trans_output(config)


def pre_train(config, run_mode, train_dataset):
    """if train process dataset"""
    if run_mode in ['train', 'finetune']:
        if train_dataset:
            config.train_dataset.data_loader.dataset_dir = train_dataset
        train_dataset = config.train_dataset.data_loader.dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='predict_qwen2_72b_instruct.yaml', type=str,
                        help='config file path.')
    parser.add_argument('--run_mode', default='predict', type=str,
                        help='set run mode for model.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model')
    parser.add_argument('--merges_file', default=None, type=str,
                        help='tokenizer model')
    parser.add_argument('--predict_data', default='', type=str,
                        help='input predict data.')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq_length')
    parser.add_argument('--predict_length', default=8192, type=int,
                        help='max length for predict output.')
    parser.add_argument('--use_parallel', default=None, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--device_id', default=-1, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    parser.add_argument('--use_past', default=None, type=str2bool,
                        help='use_past')
    parser.add_argument('--do_sample', default=None, type=str2bool,
                        help='do_sample')
    parser.add_argument('--top_k', default=None, type=int,
                        help='top_k')
    parser.add_argument('--top_p', default=None, type=float,
                        help='top_p')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--remote_save_url', default=None, type=str,
                        help='remote save url.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')

    args = parser.parse_args()
    print(args)

    if args.device_id == -1:
        args.device_id = int(os.getenv("RANK_ID", "0"))

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         use_parallel=args.use_parallel,
         use_past=args.use_past,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         vocab_file=args.vocab_file,
         merges_file=args.merges_file,
         predict_data=args.predict_data,
         train_dataset=args.train_dataset,
         seq_length=args.seq_length,
         max_length=args.predict_length,
         device_id=args.device_id,
         do_sample=args.do_sample,
         top_k=args.top_k,
         top_p=args.top_p,
         remote_save_url=args.remote_save_url,
         batch_size=args.batch_size)
