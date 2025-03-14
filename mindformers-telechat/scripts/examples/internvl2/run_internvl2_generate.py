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
"""Internvl2 predict example."""
import argparse
import os
from mindspore import Model
from mindformers import MindFormerConfig, MindFormerModuleType, MindFormerRegister
from mindformers.core.context import build_context
from mindformers.models.build_processor import build_processor
from mindformers.models.build_model import build_network
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from research.internvl2.internvl import InternVLChatModel
from research.internvl2.internvl2_tokenizer import Internvl2Tokenizer
from research.internvl2.intern_clip_vit import InternVisionPreTrainedModel, InternVisionModel
from research.internvl2.internvl2_processor import InternVLImageContentTransformTemplate
from research.internvl2.internvl_configuration import InternVisionConfig, InternVLChatConfig


def register_modules():
    MindFormerRegister.register_cls(InternVLChatConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(InternVisionConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(InternVLChatModel, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(Internvl2Tokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(InternVLImageContentTransformTemplate, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(InternVisionPreTrainedModel, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(InternVisionModel, MindFormerModuleType.MODELS)


def prepare_env(config_path):
    """Prepare environment for predict """
    config = MindFormerConfig(config_path)
    device_num = os.getenv('MS_WORKER_NUM')
    print(f"Use device number: {device_num}, it will override config.model_parallel.")
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1

    # init context
    build_context(config)
    build_parallel_config(config)
    return config


def prepare_data(config, data_path, data_mode='single'):
    """Prepare data for predict """
    inputs = []
    if not data_path:
        inputs = [[
            {"image": "/path/to/examples_image1.jpg"},
            {"text": "Please describe the image shortly."}
        ]]
    if data_mode == 'single':
        batch_size = 1
    else:
        batch_size = len(inputs)
    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    config.model.model_config.checkpoint_name_or_path = None
    return config, inputs


def infer_process(network, infer_data, processor):
    outputs = network.generate(**infer_data)
    for _, output_ids in enumerate(outputs):
        result = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        print(result)


def main(config_path, data_mode='single'):
    config = prepare_env(config_path)
    config, infer_data = prepare_data(
        config,
        data_path='',
        data_mode=data_mode
    )

    # build processor
    processor = build_processor(config.processor)
    results_ = processor([infer_data[0]])

    # build model
    network = build_network(config.model)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        infer_data_ = network.prepare_inputs_for_predict_layout(**results_)
        transform_and_load_checkpoint(config, model, network, infer_data_, do_predict=True)

    for _ in range(3):
        if data_mode == 'single':
            for sample in infer_data:
                results = processor([sample])
                infer_process(network, results, processor)
        else:
            results = processor(infer_data)
            infer_process(network, results, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_internvl2_40b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--data_mode', type=str,
                        help='if run model prediction in parallel mode.')
    args = parser.parse_args()

    main(args.config_path, args.data_mode)
