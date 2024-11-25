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
import os
import argparse
import json
import copy
from typing import Optional, Union, List, Dict
import mindspore as ms
from mindspore import Model, Tensor
from mindspore.common import initializer

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

def main():
    """main function."""
    input_questions = ["生抽和老抽的区别是什么？"]
    if args.input_file:
        input_questions = []
        input_file = open(args.input_file, 'r', encoding='utf-8')
        for line in input_file.readlines():
            dic = json.loads(line)
            input_questions.append(dic["input"])
        input_file.close()
    # set model config
    config = MindFormerConfig(args.yaml_file)
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = 'InferenceMatmulSplit,PagedAttention'

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
    tokenizer = TelechatTokenizer(config.processor.tokenizer.vocab_file, fast_tokenizer=True, trust_remote_code=True)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.batch_size = 1
    model_config.use_past = args.use_past
    model_config.use_flash_attention = True
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

    # infer
    for question in input_questions:
        logger.info(f"question : {str(question)}")
        dialog = [
            {"role": "user", "content": question}
        ]
        inputs = tokenizer.apply_chat_template(
            conversation=dialog,
            tokenize=False,
            add_generation_prompt=True
        )
        print(inputs)
        input_token = tokenizer(inputs)["input_ids"]
        print(input_token)
        outputs = model.generate(input_token,
                                 max_length=model_config.max_decode_length,
                                 do_sample=model_config.do_sample,
                                 top_k=model_config.top_k,
                                 top_p=model_config.top_p,
                                 max_new_tokens=model_config.max_new_tokens)
        response = tokenizer.decode(outputs[0][len(input_token):-1])
        logger.info(f"answer: {str(response)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str,
                        help='input to infer.')
    parser.add_argument('--vocab_file_path', default=None, type=str,
                        help='which model to use.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--auto_trans_ckpt', default=False, type=str2bool,
                        help='Auto transform load_checkpoint to load in distributed model.')
    parser.add_argument('--src_strategy_path_or_dir', default=None, type=str,
                        help='set src strategy path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default=None, type=str,
                        help='predict yaml path')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id set when run on single card. Default: 0')
    parser.add_argument('--start_token', default="<_start>", type=str,
                        help='start_token')
    parser.add_argument('--user_token', default="<_user>", type=str,
                        help='user_token')
    parser.add_argument('--bot_token', default="<_bot>", type=str,
                        help='bot_token')
    args = parser.parse_args()
    main()
