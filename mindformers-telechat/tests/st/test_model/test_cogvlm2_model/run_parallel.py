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
"""
Test module for testing the paralleled llama2 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_llama2_model/test_parallel.py
"""
import argparse

import mindspore as ms
from mindformers import build_context

from tests.utils.model_tester import ModelTester
from base_model import get_para_config, get_model
from test_train import get_dataset

ms.set_context(mode=ms.GRAPH_MODE)


def parallel_train_mp2_pp2():
    """test llama2 train in model_parallel=2, pipeline_stage=2."""
    # dp=1, mp=2, pp=2, micro_batch_num=4, micro_batch_interleave_num=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 1,
        'pipeline_stage': 2,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': False  # if set True, cause error
    }

    checker_config = {
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }

    runner = ModelTester(run_mode='train', batch_size=4, experiment_mode=True, **parallel_config)
    build_context(runner.args)
    model_config = get_para_config()
    # setting use_past = True in training will cause error
    model_config.use_past = False
    model_config.batch_size = 4
    model_config.llm_model.model_config.use_past = False
    model_config.is_dynamic = False
    model_config.llm_model.model_config.is_dynamic = False
    model_config.parallel_config = runner.args.get_parallel_config()
    model_config.parallel_config.pipeline_stage = 2
    model_config.parallel_config.data_parallel = 2
    model_config.llm_model.model_config.parallel_config = runner.args.get_parallel_config()
    model_config.vision_model.model_config.parallel_config = runner.args.get_parallel_config()

    model = get_model(model_config)


    dataset = get_dataset(seq_len=model_config.llm_model.model_config.seq_length,
                          vocab_size=model_config.llm_model.model_config.vocab_size, batch_size=4)

    runner.set_train(model, model_config, dataset=dataset, task='multi_modal_to_text_generation',
                     checker_config=checker_config)


TEST_MAP = {
    'parallel_train_mp2_pp2': parallel_train_mp2_pp2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of cogvlm2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
