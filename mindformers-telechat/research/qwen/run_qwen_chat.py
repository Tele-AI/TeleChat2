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
"""Script to run chat on Qwen-7B-Chat/Qwen-14B-Chat model."""
import os

from mindformers.core.context import build_context
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import str2bool

from qwen_config import QwenConfig
from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer
from qwen_chat import chat


def init_model(args):
    """Initialize Qwen model."""
    yaml_path = os.path.expanduser(args.config)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    config = MindFormerConfig(yaml_path)

    if args.vocab_file:
        config.processor.tokenizer.vocab_file = args.vocab_file
    vocab_file = config.processor.tokenizer.vocab_file
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(vocab_file)
    tokenizer = QwenTokenizer(**config.processor.tokenizer)

    if args.use_parallel is not None:
        config.use_parallel = args.use_parallel
    if args.device_id is not None:
        config.context.device_id = args.device_id
    if args.auto_trans_ckpt is not None:
        config.auto_trans_ckpt = args.auto_trans_ckpt

    # init context
    build_context(config)

    model_config = QwenConfig.from_pretrained(yaml_path)
    if args.seq_length:
        model_config.seq_length = args.seq_length
    if args.load_checkpoint:
        model_config.checkpoint_name_or_path = args.load_checkpoint
    if args.do_sample is not None:
        model_config.do_sample = args.do_sample

    model = QwenForCausalLM(model_config)

    if config.model.model_config.pet_config:
        print("----------------Init lora params----------------")
        pet_config = LoraConfig(
            lora_rank=config.model.model_config.pet_config.lora_rank,
            lora_alpha=config.model.model_config.pet_config.lora_alpha,
            lora_dropout=config.model.model_config.pet_config.lora_dropout,
            target_modules=config.model.model_config.pet_config.target_modules
        )
        model = get_pet_model(model, pet_config)

    return tokenizer, model, model_config


def run_chat_demo(model, tokenizer, verbose=True):
    """Run demo dialogs for Qwen-chat."""
    history = None

    query = '你好'
    response, history = chat(model, tokenizer, query, history, verbose=verbose)
    print(response)

    # prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' \
    #    + '<|im_start|>user\n你好<|im_end|>\n' \
    #    + '<|im_start|>assistant\n'

    query = '给我讲一个年轻人奋斗创业最终取得成功的故事。'
    response, history = chat(model, tokenizer, query, history, verbose=verbose)
    # prompt += '你好！很高兴为你提供帮助。<|im_end|>\n' \
    #    + '<|im_start|>user\n给我讲一个年轻人奋斗创业最终取得成功的故事。<|im_end|>\n' \
    #    + '<|im_start|>assistant\n'
    print(response)

    query = '给这个故事起一个标题'
    response, history = chat(model, tokenizer, query, history, verbose=verbose)
    print(response)


def main(args):
    """Main function."""
    tokenizer, model, _ = init_model(args)

    if args.run_demo:
        run_chat_demo(model, tokenizer)

    history = []
    if args.predict_data:
        for query in args.predict_data:
            response, history = chat(model, tokenizer, query, history,
                                     verbose=args.verbose, append_history=args.enable_history)
            print(response)
    else:
        while True:
            query = input("Input your query (type '/clear' to clear history, '/quit' to quit)\n> ")
            if query == '/clear':
                history.clear()
                continue
            elif query in ('/quit', '/exit'):
                return

            response, history = chat(model, tokenizer, query, history,
                                     verbose=args.verbose, append_history=args.enable_history)
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
                        help='input predict data (multiple values allowed, delimited by space char.')
    parser.add_argument('--enable_history', default=None, type=str2bool,
                        help='whether to enable chat history')
    parser.add_argument('--verbose', default=False, type=str2bool,
                        help='whether to print debug message when encoding/decoding chatml')
    parser.add_argument('--run_demo', default=False, type=str2bool,
                        help='run chat demo at startup')
    args_ = parser.parse_args()

    if args_.device_id == -1:
        args_.device_id = int(os.getenv("DEVICE_ID", "0"))

    main(args_)
