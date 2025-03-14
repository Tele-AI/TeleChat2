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
"""Telechat predict scripts."""
import argparse
import json
import copy
from typing import Optional, Union, List, Dict
import mindspore as ms
from mindspore import Model, Tensor
from mindspore.common import initializer

from telechat_tokenizer_fast import TelechatTokenizerFast

from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger
from mindformers.generation import GenerationConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from telechat_tokenizer import TelechatTokenizer
from telechat_config import TelechatConfig
from telechat_predict_utils import History
from telechat import TelechatForCausalLM


def chat(model, tokenizer, question: str = '', history: Union[List[Dict], History] = None,
         generation_config: Optional[GenerationConfig] = None):
    """
    Args:
        tokenizer:  the tokenizer of  telechat
        question: question which the model reply in this turn
        history: history which will format the input for telechat
        stream: if return the full text at last or yield the text in token
        generation_config:  configuration for generation
        **kwargs: args which will update the generation config or pass to model forward
    """
    if not generation_config:
        logger.error("generation_config is None")
        raise ValueError("generation_config must not be None")
    if not question:
        logger.error("question is empty")
        raise ValueError("question must not be empty")
    if history is None:
        history = []

    generation_config = copy.deepcopy(generation_config)

    # transfer to History
    if not isinstance(history, History):
        history = History(tokenizer, history)

    inputs = build_inputs_for_chat(tokenizer, question, history, generation_config)
    logger.info(f"inputs: {inputs}")
    history.append({"role": "user", "content": question})
    outputs = model.generate(inputs,
                             max_length=generation_config.max_decode_length,
                             do_sample=generation_config.do_sample,
                             top_k=generation_config.top_k,
                             top_p=generation_config.top_p,
                             max_new_tokens=generation_config.max_new_tokens)
    response = tokenizer.decode(outputs[0][len(inputs):-1])
    history.append({"role": "bot", "content": response})
    return response, history


def build_inputs_for_chat(tokenizer, question, history, generation_config):
    """
    check history and  build inputs here
    """
    # first tokenize question
    q_token = tokenizer(question)
    qa_history = copy.deepcopy(history)

    # get the max length we should build our inputs in
    model_max_length = generation_config.seq_length
    build_max_length = max(0, model_max_length - generation_config.max_new_tokens) \
        if generation_config.max_new_tokens else max(0, generation_config.max_decode_length)
    if build_max_length < 3:
        raise ValueError("the model can not meet the  requirements of input length,Please check config")

    # trunc left
    start_id = generation_config.start_token_id
    usr_id = generation_config.user_token_id
    bot_id = generation_config.bot_token_id
    input_tokens = [usr_id] + q_token["input_ids"][-build_max_length + 1:] + [bot_id]
    length = len(input_tokens)

    while len(qa_history) >= 1:
        message = qa_history.pop()
        if message["role"] == "user":
            tokens = [usr_id] + message["input_ids"]
        elif message["role"] == "bot":
            tokens = [bot_id] + message["input_ids"] + [generation_config.eos_token_id]
        else:
            tokens = []
        if len(tokens) + length >= build_max_length:
            break
        else:
            input_tokens = tokens + input_tokens
    if start_id:
        input_tokens = [start_id] + input_tokens
    return input_tokens


def main():
    """main function."""
    input_questions = []
    input_file = open(args.input_file, 'r', encoding='utf-8')
    for line in input_file.readlines():
        dic = json.loads(line)
        input_questions.append(dic["input"])
    input_file.close()

    # set model config
    config = MindFormerConfig(args.yaml_file)
    if args.device_id is not None:
        config.context.device_id = args.device_id
    if args.checkpoint_path is not None:
        config.load_checkpoint = args.checkpoint_path
    if args.use_parallel is not None:
        config.use_parallel = args.use_parallel
    if args.auto_trans_ckpt is not None:
        config.auto_trans_ckpt = args.auto_trans_ckpt
    if args.src_strategy_path_or_dir is not None:
        config.src_strategy_path_or_dir = args.src_strategy_path_or_dir
    if args.vocab_file_path is not None:
        config.processor.tokenizer.vocab_file = args.vocab_file_path

    # 初始化环境
    build_context(config)
    build_parallel_config(config)

    # build tokenizer
    if not config.processor.tokenizer.vocab_file.endswith(".json"):
        tokenizer = TelechatTokenizer(config.processor.tokenizer.vocab_file, fast_tokenizer=True)
    else:
        tokenizer = TelechatTokenizerFast(tokenizer_file=config.processor.tokenizer.vocab_file, fast_tokenizer=True)

    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.batch_size = 1
    model_config.use_past = args.use_past
    model_config.use_flash_attention = True
    model_config.start_token_id = tokenizer.convert_tokens_to_ids(args.start_token) \
        if args.start_token else None
    model_config.user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    model_config.bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    model_config.max_new_tokens = None

    model_config = TelechatConfig(**model_config)

    # build model from config
    model = TelechatForCausalLM(model_config)
    ms_model = Model(model)
    logger.info(f"[INFO_config]: {model_config}")
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = model.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, ms_model, model, infer_data, do_predict=True)
    history = []
    for question in input_questions:
        logger.info(f"question : {str(question)}")
        answer, history = chat(model, tokenizer, question, history, generation_config=model_config)
        logger.info(f"answer:, {str(answer)}")
        logger.info(f"\n截至目前的聊天记录是:, {str(history)}")
        logger.info("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str,
                        help='input to infer.')
    parser.add_argument('--yaml_file', default="predict_telechat_7b.yaml", type=str,
                        help='predict yaml path')
    parser.add_argument('--vocab_file_path', default='', type=str,
                        help='which model to use.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='Auto transform load_checkpoint to load in distributed model.')
    parser.add_argument('--src_strategy_path_or_dir', default=None, type=str,
                        help='set src strategy path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id set when run on single card. Default: 0')
    parser.add_argument('--start_token', default=None, type=str,
                        help='start_token')
    parser.add_argument('--user_token', default="<_user>", type=str,
                        help='user_token')
    parser.add_argument('--bot_token', default="<_bot>", type=str,
                        help='bot_token')
    args = parser.parse_args()
    main()
