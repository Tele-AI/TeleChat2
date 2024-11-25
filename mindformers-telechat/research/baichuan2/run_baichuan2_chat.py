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
"""Baichuan13b chat scripts."""
import os
import argparse

import mindspore as ms
from mindspore import Model
from mindspore.common import initializer as init

from mindformers import LlamaConfig
from mindformers import MindFormerConfig
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.pet import get_pet_model, LoraConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

USER_TOKEN_ID = 195
ASSISTANT_TOKEN_ID = 196

model_dict = {
    "baichuan2_7b": Baichuan7BV2ForCausalLM,
    "baichuan2_7b_lora": Baichuan7BV2ForCausalLM,
    "baichuan2_13b": Baichuan13BV2ForCausalLM,
    "baichuan2_13b_lora": Baichuan13BV2ForCausalLM
}


def main(config='./',
         use_parallel=None,
         device_id=None,
         ckpt=None,
         strategy=None,
         auto_trans_ckpt=None,
         vocab_file=None,
         max_new_tokens=512,
         use_past=True,
         do_sample=True,
         top_k=3,
         top_p=0.85,
         repetition_penalty=1.05,
         temperature=1.0):
    """main function."""

    config = MindFormerConfig(config)

    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id
    if ckpt is not None:
        if not os.path.exists(ckpt):
            raise ValueError(f"{ckpt} is not found!")
        if os.path.isfile(ckpt):
            config.load_checkpoint = None
            config.model.model_config.checkpoint_name_or_path = ckpt
        elif os.path.isdir(ckpt):
            config.load_checkpoint = ckpt
            config.model.model_config.checkpoint_name_or_path = None
    if strategy is not None and os.path.exists(strategy):
        config.src_strategy_path_or_dir = strategy
    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
    if vocab_file is None:
        if config.processor.tokenizer.vocab_file is None:
            raise ValueError("vocab_file can't be None.")
        vocab_file = config.processor.tokenizer.vocab_file
    if use_past is not None:
        config.model.model_config.use_past = use_past

    build_context(config)
    build_parallel_config(config)
    logger.info("context config is: %s", config.parallel_config)
    logger.info("moe config is: %s", config.moe_config)

    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    if parallel_mode not in ["semi_auto_parallel", "auto_parallel"]:
        config.parallel_config.data_parallel = 1
        config.parallel_config.model_parallel = 1
        config.parallel_config.pipeline_stage = 1
        config.parallel_config.micro_batch_num = 1
        logger.info("parallel_config will be change to default config: %s.",
                    config.parallel_config)

    # init tokenizer
    tokenizer = Baichuan2Tokenizer(vocab_file=vocab_file)

    # init model
    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = 1
    model_config = LlamaConfig(**config.model.model_config)
    model_config.use_flash_attention = False
    model_name = config.trainer.model_name
    network = model_dict.get(model_name)(model_config)

    if config.model.model_config.pet_config:
        logger.info("----------------Init lora params----------------")
        pet_config = LoraConfig(
            lora_rank=config.model.model_config.pet_config.lora_rank,
            lora_alpha=config.model.model_config.pet_config.lora_alpha,
            lora_dropout=config.model.model_config.pet_config.lora_dropout,
            target_modules=config.model.model_config.pet_config.target_modules
        )
        network = get_pet_model(network, pet_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("------------Transform and Load checkpoint------------")
        if ms.context.get_auto_parallel_context('parallel_mode') in \
                ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
            seq_length = config.model.model_config.seq_length
            infer_data = ms.Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
            transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)
        else:
            transform_and_load_checkpoint(config, model, network, None, do_predict=True)

    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    streamer = None

    messages = []
    while True:
        messages.append({"role": "user", "content": input("请输入：")})  # "帮助我制定一份去上海的旅游攻略"
        input_ids = build_chat_input(model_config, tokenizer, messages, max_new_tokens)
        outputs = network.generate(input_ids,
                                   max_new_tokens=max_new_tokens,
                                   streamer=streamer,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   do_sample=do_sample)
        response = tokenizer.decode(outputs[0][len(input_ids):], skip_speical_tokens=True)
        logger.info(response)
        messages.append({"role": "assistant", "content": response})
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open("question_answer.txt", flags_, 0o750), 'a') as file:
            question = messages[-2]["content"]
            answer = messages[-1]["content"]
            file.write(f"Q: {question}\n")
            file.write(f"A: {answer}\n")


def build_chat_input(config, tokenizer, messages, max_new_tokens=None):
    """add prompt for baichuan input, and truncate input if too long."""

    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        r = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                if i != 0:
                    raise ValueError(f"i should be 0, but got {i}")
                system = message["content"]
                continue
            if message["role"] == split_role and r:
                rounds.append(r)
                r = []
            r.append(message)
        if r:
            rounds.append(r)
        return system, rounds

    max_new_tokens = max_new_tokens or config.max_decode_length // 2
    max_input_tokens = config.max_decode_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if message["role"] == "user":
                round_tokens.append(USER_TOKEN_ID)
            else:
                round_tokens.append(ASSISTANT_TOKEN_ID)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if not history_tokens or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(ASSISTANT_TOKEN_ID)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="baichuan2/run_baichuan2_7b.yaml", type=str,
                        help='config of task')
    parser.add_argument('--use_parallel', default=None, type=str2bool,
                        help='use parallel')
    parser.add_argument('--device_id', default=None, type=int,
                        help='device id')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--src_strategy', default=None, type=str,
                        help='strategy of load_checkpoint')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='auto trans ckpt')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model to load.')
    parser.add_argument('--max_new_tokens', default=512, type=int,
                        help='max new tokens will be generated.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='increment predict')
    parser.add_argument('--do_sample', default=False, type=str2bool,
                        help='do sample')
    parser.add_argument('--top_k', default=1, type=int,
                        help='top k')
    parser.add_argument('--top_p', default=1.0, type=float,
                        help='top p')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help='repetition penalty')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature')
    args = parser.parse_args()

    main(config=args.config,
         use_parallel=args.use_parallel,
         device_id=args.device_id,
         ckpt=args.load_checkpoint,
         strategy=args.src_strategy,
         auto_trans_ckpt=args.auto_trans_ckpt,
         vocab_file=args.vocab_file,
         max_new_tokens=args.max_new_tokens,
         use_past=args.use_past,
         do_sample=args.do_sample,
         top_k=args.top_k,
         top_p=args.top_p,
         repetition_penalty=args.repetition_penalty,
         temperature=args.temperature)
