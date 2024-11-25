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
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig

from tests.utils.model_tester import ModelTester
from base_model import get_config, get_model

ms.set_context(mode=ms.GRAPH_MODE)


def parallel_train_mp2_pp2():
    """test llama2 train in model_parallel=2, pipeline_stage=2."""
    # dp=1, mp=2, pp=2, micro_batch_num=4, micro_batch_interleave_num=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': False  # if set True, cause error
    }
    runner = ModelTester(run_mode='train', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.631487, 10.624989, 10.625435, 10.618025, 10.612650,
                10.614664, 10.616899, 10.601905, 10.601352, 10.595778,
                10.593061, 10.595354, 10.585533, 10.581073, 10.585693,
                10.581120, 10.580448, 10.573355, 10.584267, 10.586115]
    time_std = 550

    checker_config = {
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


def parallel_train_dp2():
    """test llama2 train in data_parallel=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }
    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.629693, 10.624046, 10.625021, 10.609780, 10.617902,
                10.622823, 10.612687, 10.619169, 10.621492, 10.612216,
                10.604536, 10.601242, 10.589621, 10.596098, 10.601740,
                10.572743, 10.585052, 10.599199, 10.602436, 10.577181]
    time_std = 280

    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std)


def parallel_predict_mp2():
    """test llama2 predict in model_parallel=2."""
    # dp=1, mp=4, pp=1
    parallel_config = {
        'use_parallel': True,
        'model_parallel': 2,
        'use_seq_parallel': False
    }
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    runner = ModelTester(run_mode='predict', batch_size=2, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model_config.batch_size = runner.batch_size  # set batch size for prediction

    model = get_model(model_config)

    outputs = r"hello world.ýedeshapeenty BritainAfter przyowy französ夏 très befindethape정dot programa"
    runner.set_predict(model=model, expect_outputs=outputs)


def parallel_train_mp2_cp2():
    """test llama2 train in model_parallel=2, context_parallel=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'use_seq_parallel': False,
        'enable_parallel_optimizer': True
    }
    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    # only support mp with cp
    model_config.parallel_config = TransformerOpParallelConfig(model_parallel=2,
                                                               context_parallel=2,
                                                               use_seq_parallel=False)
    model = get_model(model_config)

    loss_std = [10.627002, 10.620553, 10.613135, 10.614858, 10.623809,
                10.623188, 10.600923, 10.629244, 10.610760, 10.610001,
                10.600121, 10.607943, 10.610308, 10.596238, 10.597507,
                10.595253, 10.594837, 10.600836, 10.585949, 10.604373]
    time_std = 400
    context_parallel = {
        'data_parallel': 1,
        'model_parallel': 2,
        'context_parallel': 2,
        'pipeline_stage': 1,
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     parallel_config=context_parallel)


def parallel_train_sapp_mp2_pp2():
    """test llama2 train in auto_parallel and model_parallel=2, pipeline_stage=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2,
        'enable_parallel_optimizer': True,
        'parallel_mode': 2,  # ms.ParallelMode.AUTO_PARALLEL
        'search_mode': "recursive_programming"
    }
    runner = ModelTester(run_mode='train', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.631487, 10.624983, 10.625433, 10.618023, 10.612650,
                10.614660, 10.616902, 10.601908, 10.601349, 10.595780,
                10.593057, 10.595364, 10.585537, 10.581072, 10.585690,
                10.581121, 10.580455, 10.573353, 10.584270, 10.586113]
    time_std = 600

    checker_config = {
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


TEST_MAP = {
    'parallel_train_mp2_pp2': parallel_train_mp2_pp2,
    'parallel_train_dp2': parallel_train_dp2,
    'parallel_predict_mp2': parallel_predict_mp2,
    'parallel_train_mp2_cp2': parallel_train_mp2_cp2,
    'parallel_train_sapp_mp2_pp2': parallel_train_sapp_mp2_pp2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
