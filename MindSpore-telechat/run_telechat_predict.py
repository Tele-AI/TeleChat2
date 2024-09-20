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

from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.tools import set_output_path
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger
from mindformers.generation import GenerationConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from telechat_tokenizer import TelechatTokenizer
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
from research.telechat.telechat_predict_utils import History


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
    user_id = generation_config.user_token_id
    bot_id = generation_config.bot_token_id

    # transfer to History
    if not isinstance(history, History):
        history = History(tokenizer, history)

    inputs = build_inputs_for_chat(tokenizer, question, history, generation_config, user_id, bot_id)
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


def build_inputs_for_chat(tokenizer, question, history, generation_config, usr_id, bot_id):
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
    config.context.device_id = 0
    if args.checkpoint_path:
        config.load_checkpoint = args.checkpoint_path
    config.use_parallel = True
    # 初始化环境
    set_output_path(args.output_chat)
    build_context(config)
    build_parallel_config(config)

    # build tokenizer
    tokenizer = TelechatTokenizer(args.vocab_file_path, fast_tokenizer=True, trust_remote_code=True)

    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.batch_size = 1
    model_config.run_mode = "predict"
    model_config.use_past = args.use_past
    model_config.use_flash_attention = True
    model_config.user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    model_config.bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    model_config.max_new_tokens = None

    model_config = TelechatConfig(**model_config)

    # build model from config
    model = TelechatForCausalLM(model_config)
    ms_model = Model(model)
    print(f"[INFO_config]: {model_config}")
    print("----------------Transform and load checkpoint----------------")
    seq_length = model_config.seq_length
    input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
    infer_data = model.prepare_inputs_for_predict_layout(input_ids)
    transform_and_load_checkpoint(config, ms_model, model, infer_data, do_predict=True)
    history = []
    for question in input_questions:
        print("question:", question)
        answer, history = chat(model, tokenizer, question, history, generation_config=model_config)
        print("answer:", answer)
        print("\n截至目前的聊天记录是:", history)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str,
                        help='input to infer.')
    parser.add_argument('--vocab_file_path', default='', type=str,
                        help='which model to use.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--output_chat', default="./output_chat", type=str,
                        help='chat output')
    parser.add_argument('--user_token', default="<_user>", type=str,
                        help='user_token')
    parser.add_argument('--bot_token', default="<_bot>", type=str,
                        help='bot_token')
    args = parser.parse_args()
    main()
