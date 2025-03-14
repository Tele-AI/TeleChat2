# Copyright 2025 Huawei Technologies Co., Ltd
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
""" Run SeqPipeline Parallel """
import os
import sys
import argparse
from mindformers.tools.register import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.trainer import Trainer
import numpy as np
import mindspore as ms
from mindspore import Tensor

ms.set_seed(2025)
np.random.seed(2025)

class FakeData:
    """ generate fake data for pipeline parallel test """
    def __init__(self, data_num, seq_length, input_data=None):
        super().__init__()
        if input_data is not None:
            self.input_data = input_data
            self.data_num = self.input_data.shape[0]
            self.seq_length = self.input_data[0].shape[0]
        else:
            self.input_data = np.random.randint(0, 100, (data_num, seq_length+1))

    def __getitem__(self, index):
        return Tensor(self.input_data[index], dtype=ms.int32)

    def __len__(self):
        return self.input_data.shape[0]

def main(config, dataset):
    """main."""
    # init context
    build_context(config)

    trainer = Trainer(config, train_dataset=dataset)
    trainer.train()


if __name__ == '__main__':
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="./finetune_llama3_8b.yaml",
        required=True,
        help='YAML config files')
    parser.add_argument(
        '--run_mode', default='train', type=str,
        help='task running status, it support [train, finetune, eval, predict].'
             'Default: None')
    parser.add_argument(
        '--register_path', default=None, type=str,
        help='the register path of outer API.')
    args_, rest_args_ = parser.parse_known_args()

    if args_.register_path is not None:
        if not os.path.isabs(args_.register_path):
            args_.register_path = os.path.abspath(args_.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = args_.register_path
        if args_.register_path not in sys.path:
            sys.path.insert(0, args_.register_path)

    config_ = MindFormerConfig(args_.config, run_mode=args_.run_mode)
    config_.use_parallel = True

    fake_dataset = FakeData(data_num=32, seq_length=config_.model.model_config.seq_length)
    main(config_, fake_dataset)
