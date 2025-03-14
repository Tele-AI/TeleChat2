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
"""convert huggingface safetensors"""
import argparse
import os
import time

from mindformers.tools import MindFormerConfig, MindFormerRegister, MindFormerModuleType, logger
from mindformers.utils import convert_hf_safetensors_multiprocess

if __name__ == '__main__':
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str,
                        required=True,
                        help='A directory of HuggingFace safetensors. '
                             'Usually a directory containing HuggingFace\'s safetensors file. '
                             'If they are in the form of a slice, '
                             'it also needs to contain the \'model.safetensors.index.json\' file.')
    parser.add_argument('--dst_dir',
                        default="./transformed_output",
                        type=str,
                        help='A directory to save transformed safetensors. '
                             'If the given directory does not exist, it is created automatically. '
                             'If the given directory already exists, it will overwrite its contents.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='The path of model YAML file.')
    parser.add_argument('--register_path',
                        type=str,
                        default=None,
                        help='The register path of outer API. '
                             'This is usually the parent path of the Python file where the outer API is located.'
                             'This configuration can be ignored if it does not involve registering the outer API.')
    args = parser.parse_args()

    if args.register_path is not None:
        if not os.path.isabs(args.register_path):
            args.register_path = os.path.join(work_path, args.register_path)
        os.environ["REGISTER_PATH"] = args.register_path

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    model = MindFormerConfig(args.config).model
    if 'auto_register' in model:
        MindFormerRegister.auto_register(class_reference=model.pop('auto_register'),
                                         module_type=MindFormerModuleType.MODELS)
    model_cls = MindFormerRegister.get_cls(MindFormerModuleType.MODELS, model.arch.type)

    logger.info(f"src_dir: {src_dir}")
    logger.info(f"dst_dir: {dst_dir}")
    logger.info(f"dst_dir: {dst_dir}")

    start_time = time.time()
    convert_hf_safetensors_multiprocess(src_dir, dst_dir, model_cls, model.model_config)
    cost_time = time.time() - start_time
    logger.info(f"Convert safetensors cost: {cost_time}s")
