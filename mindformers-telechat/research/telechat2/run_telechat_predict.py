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
import mindspore as ms
from mindspore import Model, Tensor
from mindspore.common import initializer

from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from research.telechat2.telechat_tokenizer import TelechatTokenizer
from research.telechat2.telechat_config import TelechatConfig
from research.telechat2.infer.telechat import ParallelTelechatForCausalLM


def main():
    """main function."""
    input_questions = ["生抽与老抽的区别？"]

    # set config
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

    # init context
    build_context(config)
    build_parallel_config(config)

    # build tokenizer
    chat_template = "{%- if tools %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{-'<_system>'+messages[0]['content'] }}\n    {%- else %}\n        {{- '<_system>'+'你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。' }}\n    {%- endif %}\n    {{- '\\n\\n# 可用工具\\n你可以调用<tools></tools>标签中包含的一个或多个工具来辅助你回答问题,以下是可用工具详情：\\n<tools>\\n' }}\n    {%- for tool in tools %}\n        {{- tool | tojson }}\n        {{-'\\n'}}\n    {%- endfor %}\n    {{- '</tools>\\n\\n# 调用方法\\n你需要遵循工具的要求，使用json格式返回工具名称及参数，并用<tool_call></tool_call>包含。下方是一个调用模板：\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>\\n\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<_system>' + messages[0]['content'] + '\\n' }}\n    {%- else %}\n        {{- '<_system>'+'你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') %}\n        {{- '<_user>' + message.content }}\n    {%- elif message.role == 'bot' or message.role == 'assistant' %}\n        {{- '<_bot>' }}\n        {%- if message.content %}\n            {{- message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {%- if loop.index0 == 0 %}\n                {{-'<tool_call>'}}\n            {%- else %}\n                {{-'\\n<tool_call>'}}\n            {%- endif %}\n            {{- '\\n{\"name\": \"' }}{{ tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<_end>\\n' }}\n    {%- elif message.role == 'tool' %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n            {{- '<_user>'+'<tool_response>\\n' }}\n        {%- else %}\n            {{- '\\n<tool_response>\\n' }}\n        {%- endif %}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<_bot>' }}\n{%- endif %}"
    tokenizer = TelechatTokenizer(config.processor.tokenizer.vocab_file, \
        chat_template=chat_template, fast_tokenizer=True, trust_remote_code=True)

    # build model config
    model_config = config.model.model_config
    model_config.moe_config = config.moe_config
    model_config.parallel_config = config.parallel_config
    model_config.batch_size = 1
    model_config.use_past = args.use_past
    model_config.use_flash_attention = True
    model_config.max_length = args.max_length
    model_config.max_new_tokens = args.max_new_tokens
    model_config.do_sample = args.do_sample
    model_config.top_k = args.top_k
    model_config.top_p = args.top_p
    model_config.repetition_penalty = args.repetition_penalty
    model_config = TelechatConfig(**model_config)

    # build model from config
    model = ParallelTelechatForCausalLM(model_config)
    ms_model = Model(model)
    logger.info(f"[INFO_config]: {model_config}")
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = model.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, ms_model, model, infer_data, do_predict=True)

    inputs = []
    for question in input_questions:
        inputs.append({"role": "user", "content": question})
        inputs_chat = tokenizer.apply_chat_template(conversation=inputs, tokenize=False, add_generation_prompt=True)
        logger.info(f"inputs: {inputs_chat}")
        input_ids = tokenizer(inputs_chat)["input_ids"]
        logger.debug(f"input_ids: {input_ids}")
        outputs = model.generate(input_ids)
        output_ids = outputs[0][len(inputs):-1]
        logger.debug(f"output_ids: {output_ids}")
        answer = tokenizer.decode(outputs[0][len(input_ids):-1])
        logger.info(f"answer: {answer}")
        inputs.append({"role": "bot", "content": answer})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--max_new_tokens', default=512, type=int,
                        help='Maximum generation length.')
    parser.add_argument('--max_length', default=8192, type=int,
                        help='The maximum length of input plus output.')
    parser.add_argument('--do_sample', default=False, type=bool,
                        help='Enable top-k or top-p sampling.')
    parser.add_argument('--top_k', default=1, type=int,
                        help='Sample from the top-k tokens with the highest probabilities.')
    parser.add_argument('--top_p', default=1.0, type=float,
                        help='Sample from the tokens with the highest probabilities \
                            whose cumulative probabilities do not exceed top-p.')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help='The penalty coefficient for repeated token.')
    args = parser.parse_args()
    main()
