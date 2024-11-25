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
"""glm4 predict example."""
import os
import argparse

import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration, ChatGLM4Tokenizer
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint


def main(config_path, use_parallel, load_checkpoint, vocab_file):
    inputs = ["晚上睡不着应该怎么办", "使用python编写快速排序代码", "你好呀！"]
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

    os.environ["RUN_MODE"] = config.run_mode

    # init context
    build_context(config)
    build_parallel_config(config)

    # init model
    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = ChatGLM2Config(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # init tokenizer
    tokenizer = ChatGLM4Tokenizer(vocab_file=vocab_file)

    # build model
    network = ChatGLM2ForConditionalGeneration(model_config)
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

    chat_template = config.processor.tokenizer.chat_template if config.processor.tokenizer.chat_template else None

    if isinstance(inputs, list):
        inputs_ids = tokenizer.build_batch_input(inputs)["input_ids"]
    else:
        if not isinstance(inputs, str):
            raise ValueError("inputs must be a str, but got {}".format(type(inputs)))
        inputs_ids = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": inputs}],
                                                   chat_template=chat_template,
                                                   add_generation_prompt=True,
                                                   return_tensors="np")

    outputs = network.generate(inputs_ids,
                               max_length=model_config.max_decode_length,
                               do_sample=model_config.do_sample,
                               top_k=model_config.top_k,
                               top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_glm4_9b_chat.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', default=False, type=bool,
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', type=str,
                        help='tokenizer.model file path.')
    args = parser.parse_args()
    main(args.config_path, args.use_parallel, args.load_checkpoint, args.vocab_file)
