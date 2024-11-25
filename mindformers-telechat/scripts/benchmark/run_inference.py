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
"""Run inference process"""
import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.common import initializer as init

import mindformers
from mindformers import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from scripts.benchmark.base_init_model import BaseInitModel, convert_path


cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(mindformers.__file__))


class ModelPredict(BaseInitModel):
    """Inherited from BaseInitModel"""

    def __init__(self, model_name_or_dir: str = 'llama2_7b', predict_data=None, input_args=None):
        super().__init__(model_name_or_dir=model_name_or_dir, input_args=input_args)

        # prepare predict data
        data_seq_len = self.input_args.get('data_seq_len')
        data_batch_size = self.input_args.get('data_batch_size')
        if data_seq_len and data_batch_size:
            self.custom_data = True
            self.batch_size = data_batch_size
            self.inputs = self.prepare_fake_data(data_seq_len)
        else:
            self.custom_data = False
            self.inputs, self.batch_size = self.prepare_data(predict_data)

    def prepare_data(self, train_data):
        """Prepare data"""
        if not isinstance(train_data, list):
            raise ValueError("input training data should be a file path or multiple str data.")

        if len(train_data) == 1 and os.path.isfile(train_data[0]):
            with open(train_data[0], 'r') as file:
                input_data = file.readlines()
                input_data = [_.strip() for _ in input_data]
        else:
            input_data = train_data

        batch_size = len(train_data)
        return input_data, batch_size

    def prepare_fake_data(self, seq_length):
        """Prepare fake data"""
        inputs = np.random.randint((self.batch_size, seq_length), dtype=np.int32)
        return inputs.tolist()

    def process_tokenizer(self):
        """Process tokenizer"""
        if not self.config.processor.tokenizer.get('vocab_file', None):
            return
        tokenizer_files = glob(f"{self.model_path}/tokenizer.model")
        vocab_file = self.input_args.get('vocab_file', None)
        if not vocab_file or not os.path.exists(vocab_file):
            vocab_file = self.config.processor.tokenizer.vocab_file

        if vocab_file and os.path.exists(vocab_file):
            # make soft link of tokenizer file in model_path
            src_path, vocab_file = convert_path(vocab_file, self.model_path)
            try:
                os.symlink(src_path, vocab_file)
            except OSError:
                logger.warning(f"link of {vocab_file} is existed.")
        elif tokenizer_files:
            vocab_file = tokenizer_files[0]
        else:
            logger.error(f"Currently dose not support downloading the tokenizer "
                         f"and vocab file of {self.model_name} from online.")
            return
        logger.info(f"use tokenizer file: {vocab_file}.")

        self.config.processor.tokenizer.vocab_file = vocab_file

    def predict(self, generate_config=None):
        """Predict function"""
        if not self.custom_data:
            # process tokenizer
            self.process_tokenizer()
            tokenizer = build_tokenizer(self.config.processor.tokenizer)
        else:
            tokenizer = None
            logger.warning(f"use custom data, will not build tokenizer.")

        if generate_config is None:
            generate_config = self.network.generation_config
        for k, v in self.input_args.items():
            if hasattr(generate_config, k) and not callable(getattr(generate_config, k)):
                setattr(generate_config, k, v)

        if tokenizer:
            inputs_ids = tokenizer(self.inputs,
                                   max_length=self.config.model.model_config.seq_length,
                                   padding="max_length")["input_ids"]
        else:
            inputs_ids = self.inputs
        outputs = self.network.generate(inputs_ids, generation_config=generate_config)

        if tokenizer:
            for output in outputs:
                print(tokenizer.decode(output))
        else:
            print(outputs)

    def load_checkpoint(self, load_checkpoint=None):
        """Load checkpoint"""
        if load_checkpoint and os.path.exists(load_checkpoint):
            load_checkpoint = os.path.realpath(load_checkpoint)
            self.config.load_checkpoint = load_checkpoint
        elif not self.custom_model:
            # process checkpoint
            self.process_checkpoint()
        else:
            self.config.load_checkpoint = ""
            logger.warning(f"use custom model, will not load checkpoint.")

        if self.config.load_checkpoint:
            seq_length = self.config.model.model_config.seq_length
            input_ids = Tensor(shape=(self.batch_size, seq_length), dtype=ms.int32, init=init.One())
            infer_data = self.network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(self.config, self.model, self.network, infer_data, do_predict=True)


def convert_type(data):
    """Convert the type of data"""
    if data.lower() == 'true':
        return True
    if data.lower() == 'false':
        return False

    try:
        if '.' in data:
            data = float(data)
        else:
            data = int(data)
    except ValueError:
        logger.debug(f"try to convert '{data}' to float or int failed.")
    return data


def build_args(args_):
    """Build input_args"""
    error_info = "script input args should be aligned '--key value',"
    if divmod(len(args_), 2)[1] != 0:
        raise ValueError(f"{error_info} length of args is not correct.")

    input_args = dict()
    for arg_key, arg_value in zip(args_[0::2], args_[1::2]):
        if len(arg_key) <= 2 or arg_key[:2] != '--':
            raise ValueError(f"{error_info} got '{arg_key}' in keys.")
        if arg_value[:2] == '--':
            raise ValueError(f"{error_info} got '{arg_value}' in values.")

        value = convert_type(arg_value)
        input_args[arg_key[2:]] = value
    return input_args


def main(model_name_or_dir: str = 'llama2_7b', predict_data=None, input_args=None):
    if Path(model_name_or_dir).is_dir():
        yaml_files = glob(str(Path(model_name_or_dir) / '*.yaml'))
        if yaml_files:
            config = MindFormerConfig(yaml_files[0])
            build_context(config)
            build_parallel_config(config)
    else:
        raise RuntimeError('Currently only support offline infer')

    model_predict = ModelPredict(model_name_or_dir, predict_data, input_args)
    model_predict.load_checkpoint()
    model_predict.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_dir', type=str, default="llama2_7b",
        help='input model name with size, e.g., llama2_7b.')
    parser.add_argument(
        '--predict_data', metavar='N', type=str, nargs='+',
        help='multiple input predict data.')

    args, unknown_args = parser.parse_known_args()
    unknown_args = build_args(unknown_args)

    main(
        args.model_name_or_dir,
        args.predict_data,
        unknown_args
    )
