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
"""baichuan2 predict example."""
import os
import argparse
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.models import LlamaConfig
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from research.baichuan2.baichuan2_7b import Baichuan7BV2ForCausalLM
from research.baichuan2.baichuan2_13b import Baichuan13BV2ForCausalLM
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer

MODEL_MAP = {
    "baichuan2_7b": Baichuan7BV2ForCausalLM,
    "baichuan2_7b_lora": Baichuan7BV2ForCausalLM,
    "baichuan2_13b": Baichuan13BV2ForCausalLM,
    "baichuan2_13b_lora": Baichuan13BV2ForCausalLM
}


def main(config_path, use_parallel, load_checkpoint, vocab_file, predict_data):
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
    model_name = config.trainer.model_name
    config.model.model_config.parallel_config = config.parallel_config
    if config.use_parallel:
        # baichuan2 13b and 7b not support dynamic inputs in parallel
        config.model.model_config.is_dynamic = False
        logger.warning(f"{model_name} not support dynamic inputs in parallel, set is_dynamic=False Default.")
    else:
        config.model.model_config.is_dynamic = True
    config.model.model_config.use_flash_attention = False
    logger.warning(f"Flash Attention might cause accuracy issues, set use_flash_attention=False Default.")
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # init tokenizer
    tokenizer = Baichuan2Tokenizer(vocab_file=vocab_file)

    # build model
    network = MODEL_MAP[model_name](model_config)

    if config.model.model_config.pet_config:
        logger.info("----------------Init lora params----------------")
        pet_config = config.model.model_config.pet_config
        pet_config = LoraConfig(
            lora_rank=pet_config.lora_rank,
            lora_alpha=pet_config.lora_alpha,
            lora_dropout=pet_config.lora_dropout,
            target_modules=pet_config.target_modules
        )
        network = get_pet_model(network, pet_config)
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

    # predict using generate
    predict_data = f"<reserved_106>{predict_data}<reserved_107>"
    inputs_ids = tokenizer(predict_data, max_length=128, padding="max_length")["input_ids"]
    outputs = network.generate(inputs_ids,
                               do_sample=False,
                               top_k=1,
                               top_p=1.0,
                               repetition_penalty=1.0,
                               temperature=1.0,
                               max_length=128)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_baichuan2_7b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', type=str,
                        help='tokenizer.model file path.')
    parser.add_argument('--predict_data', type=str,
                        help='input data for predict.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.vocab_file,
        args.predict_data
    )
