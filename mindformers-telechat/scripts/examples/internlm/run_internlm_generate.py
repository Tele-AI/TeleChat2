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
"""internlm predict example."""
import argparse
import os

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from internlm import InternLMForCausalLM
from internlm_config import InternLMConfig
from internlm_tokenizer import InternLMTokenizer


def main(config_path, use_parallel, load_checkpoint, vocab_file):
    # multi batch inputs
    inputs = ["你是谁？",
              "你好"]
    batch_size = len(inputs)

    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = use_parallel
    device_num = os.getenv('MS_WORKER_NUM')
    logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = InternLMConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = InternLMTokenizer(vocab_file=vocab_file)

    # build model
    network = InternLMForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        # set auto transform ckpt
        if os.path.isdir(config.load_checkpoint) or config.use_parallel:
            config.auto_trans_ckpt = True
        else:
            config.auto_trans_ckpt = False
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    prompt = ["<s><|User|>:{}<eoh>\n<|Bot|>".format(input) for input in inputs]
    inputs_ids = tokenizer(prompt, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = network.generate(inputs_ids,
                               do_sample=False,
                               top_k=1,
                               top_p=1.0,
                               repetition_penalty=1.0,
                               temperature=1.0,
                               max_length=256)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', type=str,
                        help='tokenizer.model file path.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.vocab_file
    )

# 多batch输出
# <|User|>:你是谁？<eoh>\n<|Bot|>:我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我能够回答问题.....<eoa>
# <|User|>:你好<eoh>\n<|Bot|>:你好！有什么可以帮助你的吗？<eoa>
