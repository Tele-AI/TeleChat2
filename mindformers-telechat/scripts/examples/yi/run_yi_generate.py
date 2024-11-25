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
"""Yi predict example."""
import os
import argparse

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.models.llama import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config


def main(config_path, use_parallel, load_checkpoint, vocab_file, predict_mode):
    """main function."""
    inputs = ["以雷霆之力", "小明和小红"]
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

    # init model
    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None
    model_config.batch_size = batch_size

    # build tokenizer
    config.processor.tokenizer.vocab_file = vocab_file
    tokenizer = LlamaTokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
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
        input_ids = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    for prompt in inputs:
        if predict_mode == 'Chat':
            message = [{"role": "user", "content": prompt}]
            input_id = tokenizer.apply_chat_template(conversation=message,
                                                     tokenize=True,
                                                     add_generation_prompt=True)
        else:
            input_id = tokenizer.encode(prompt)

        outputs = network.generate(input_id,
                                   max_length=model_config.max_decode_length,
                                   do_sample=model_config.do_sample)
        result = tokenizer.decode(outputs[0][len(input_id):], skip_special_tokens=True)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_yi_6b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', type=str,
                        help='tokenizer.model file path.')
    parser.add_argument('--predict_mode', default="Base", type=str,
                        help="predict mode is 'Base' or 'Chat'")

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.vocab_file,
        args.predict_mode
    )
