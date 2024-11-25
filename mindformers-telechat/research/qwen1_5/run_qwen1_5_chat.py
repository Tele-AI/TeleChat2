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
"""Script to run chat on Qwen1.5-7B-Chat/Qwen1.5-14B-Chat/Qwen1.5-72B-Chat model."""
import os
import sys

import mindspore as ms
from mindspore import Model
from mindspore.common import initializer as init
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.tools import get_output_root_path
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import check_in_modelarts, str2bool
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint

from mindformers import LlamaConfig, LlamaForCausalLM
from qwen1_5_tokenizer import Qwen2Tokenizer
from qwen1_5_chat import chat


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        import moxing as mox

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
        import shutil

        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(strategy_dir)
        transformed_ckpt_dir = os.path.join(
            get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(transformed_ckpt_dir)
        os.makedirs(strategy_dir, exist_ok=True)
        os.makedirs(transformed_ckpt_dir, exist_ok=True)


def init_model(args):
    """Initialize Qwen1.5 model."""
    yaml_path = os.path.expanduser(args.config)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    config = MindFormerConfig(yaml_path)

    if args.vocab_file:
        config.processor.tokenizer.vocab_file = args.vocab_file
    vocab_file = config.processor.tokenizer.vocab_file
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(vocab_file)
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    if args.use_parallel is not None:
        config.use_parallel = args.use_parallel
    if args.device_id is not None:
        config.context.device_id = args.device_id

    # init context
    build_context(config)
    build_parallel_config(config)
    model_config = LlamaConfig.from_pretrained(yaml_path)
    model_config.parallel_config = config.parallel_config
    if args.seq_length:
        model_config.seq_length = args.seq_length
    if args.load_checkpoint:
        # we'll transform & load checkpoint later
        model_config.checkpoint_name_or_path = None
    if args.do_sample is not None:
        model_config.do_sample = args.do_sample

    network = LlamaForCausalLM(model_config)

    if config.model.model_config.pet_config:
        logger.info("----------------Init lora params----------------")
        pet_config = LoraConfig(
            lora_rank=config.model.model_config.pet_config.lora_rank,
            lora_alpha=config.model.model_config.pet_config.lora_alpha,
            lora_dropout=config.model.model_config.pet_config.lora_dropout,
            target_modules=config.model.model_config.pet_config.target_modules
        )
        network = get_pet_model(network, pet_config)

    if args.auto_trans_ckpt is not None:
        config.auto_trans_ckpt = args.auto_trans_ckpt
        if config.auto_trans_ckpt:
            clear_auto_trans_output(config)

    if args.load_checkpoint:
        config.load_checkpoint = args.load_checkpoint
        model = Model(network)
        if ms.context.get_auto_parallel_context('parallel_mode') in \
                ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
            logger.info(
                "------------Transform and Load checkpoint------------")
            seq_length = config.model.model_config.seq_length
            input_ids = ms.Tensor(shape=(1, seq_length),
                                  dtype=ms.int32, init=init.One())
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(
                config, model, network, infer_data, do_predict=True)
        else:
            transform_and_load_checkpoint(
                config, model, network, None, do_predict=True)

    return tokenizer, network, model_config


def run_chat_demo(model, model_config, tokenizer, verbose=True):
    """Run demo dialogs for Qwen-chat."""
    history = None

    query = '你好'
    print('>', query)
    response, history = chat(
        model, model_config, tokenizer, query, history, verbose=verbose)
    print(response)

    query = '给我讲一个年轻人奋斗创业最终取得成功的故事。'
    print('>', query)
    response, history = chat(
        model, model_config, tokenizer, query, history, verbose=verbose)
    print(response)

    query = '给这个故事起一个标题'
    print('>', query)
    response, history = chat(
        model, model_config, tokenizer, query, history, verbose=verbose)
    print(response)


def main(args):
    """Main function."""
    tokenizer, model, model_config = init_model(args)

    if args.run_demo:
        run_chat_demo(model, model_config, tokenizer)

    history = []
    if args.predict_data:
        for query in args.predict_data:
            print('>', query)
            response, history = chat(model, model_config, tokenizer, query, history,
                                     system=args.system_prompt,
                                     verbose=args.verbose, append_history=args.enable_history)
            print(response)
    else:
        while True:
            query = input(
                "Input your query (type '/clear' to clear history, '/quit' to quit)\n> ")
            if query == '/clear':
                history.clear()
                continue
            elif query in ('/quit', '/exit'):
                return

            response, history = chat(model, model_config, tokenizer, query, history,
                                     system=args.system_prompt,
                                     verbose=args.verbose, chat_format='chatml', append_history=args.enable_history)
            print(response)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='run_qwen_7b.yaml', type=str,
                        help='config file path (default: ./run_qwen_7b.yaml)')
    parser.add_argument('--device_id', default=-1, type=int,
                        help='ID of the target device')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--load_checkpoint', default='', type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--vocab_file', default="", type=str,
                        help='tokenizer model file.')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq_length')
    parser.add_argument('--do_sample', default=None, type=str2bool,
                        help='do_sample')
    parser.add_argument('--predict_data', default=None, type=str, nargs='+',
                        help='input predict data (multiple values allowed).')
    parser.add_argument('--system_prompt', default='You are a helpful assistant.', type=str,
                        help='system prompt used in the beginning of each chatml input.')
    parser.add_argument('--enable_history', default=None, type=str2bool,
                        help='whether to enable chat history')
    parser.add_argument('--verbose', default=False, type=str2bool,
                        help='whether to print debug message when encoding/decoding chatml')
    parser.add_argument('--run_demo', default=False, type=str2bool,
                        help='run chat demo at startup')
    args_ = parser.parse_args()

    if args_.device_id == -1:
        args_.device_id = int(os.getenv("DEVICE_ID", "0"))

    if args_.use_parallel and not args_.predict_data:
        print("Error: currently '--use_parallel' can't be used with interactivate chat.")
        print("       you need to specify inputs with '--predict_data'.")
        sys.exit(-1)

    main(args_)
