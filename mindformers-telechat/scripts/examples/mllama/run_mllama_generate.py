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
"""llama3 predict example."""
import argparse
import os
import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.models.mllama import MllamaForConditionalGeneration, MllamaConfig
from mindformers.core.context import build_context
from mindformers.models.build_processor import build_processor
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint


def main(config_path, use_parallel, load_checkpoint, vocab_file):
    # multi batch inputs
    image_path = "please set a image path."
    if not os.path.exists(image_path):
        raise ValueError(
            f"Image is not existed, please set a true path. For example, you can download "
            f"https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg "
            f"and image path to demo.jpeg path")
    text = "<|image|>Describe the image in English:"
    inputs = [{"image": image_path},
              {"text": text}]
    batch_size = 1

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
    model_config = MllamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build model
    network = MllamaForConditionalGeneration(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        # set auto transform ckpt
        config.auto_trans_ckpt = not os.path.isdir(config.load_checkpoint) and config.use_parallel
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    config.processor.tokenizer.vocab_file = vocab_file
    processor = build_processor(config.processor)
    processor_res = processor(inputs)
    for _ in range(1):
        outputs = network.generate(**processor_res,
                                   max_length=13,
                                   max_new_tokens=30,
                                   do_sample=model_config.do_sample,
                                   top_k=model_config.top_k,
                                   top_p=model_config.top_p)
        res = processor.post_process(outputs, skip_special_tokens=True)
        print(res)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_llama3_70b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.vocab_file
    )
