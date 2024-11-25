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
Test module for testing the paralleled llama interface used for mindformers.
"""
import argparse
import os

import mindspore as ms
from mindformers import build_context

from tests.utils.model_tester import ModelTester
from base_model import get_config, get_model

ms.set_context(mode=ms.GRAPH_MODE)


def _save_or_load_ckpt(model, save_ckpt: bool, load_ckpt: bool, ckpt_rel_path: str = 'ckpt/llama_single_train.ckpt'):
    """save or load checkpoint"""
    if save_ckpt and load_ckpt:
        raise ValueError("Unexpected argument value: save_ckpt and load_ckpt are both True.")

    if save_ckpt or load_ckpt:
        ckpt_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ckpt_rel_path)
        os.makedirs(os.path.dirname(ckpt_file_path), exist_ok=True)
    else:
        ckpt_file_path = None

    if save_ckpt:
        ms.save_checkpoint(model, ckpt_file_path)
    if load_ckpt:
        ms.load_checkpoint(ckpt_file_path, model)


def _base_train(args, loss_std, base_parallel_config, new_parallel_config=None):
    """base llama train"""
    if args.experiment_mode is None or args.save_ckpt is None or args.load_ckpt is None:
        raise ValueError("Unexpected argument value: any of experiment_mode or save_ckpt or load_ckpt is None.")

    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=args.experiment_mode, **base_parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    dataset = runner.get_dataset(model_config)
    if new_parallel_config is not None:
        new_parallel_config.update(**base_parallel_config)
        for key, value in new_parallel_config.items():
            model_config.__setattr__(key, value)
    model = get_model(model_config)

    _save_or_load_ckpt(model, args.save_ckpt, args.load_ckpt)
    runner.set_train(model, model_config, loss_std=loss_std, dataset=dataset)


def single_train(args):
    """test llama train in single"""
    parallel_config = {
        'use_parallel': False,
        'data_parallel': 1,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    old_loss_std = [10.644210, 10.639138, 10.638718, 10.633285, 10.619514,
                    10.636455, 10.618894, 10.630442, 10.612017, 10.634727,
                    10.624849, 10.622943, 10.633147, 10.631532, 10.634797,
                    10.627925, 10.634336, 10.638637, 10.636755, 10.631924]

    new_loss_std = [10.626314, 10.624199, 10.631677, 10.629707, 10.635541,
                    10.630877, 10.616421, 10.628949, 10.628935, 10.622899,
                    10.635184, 10.622277, 10.640517, 10.625608, 10.629306,
                    10.631349, 10.633284, 10.635038, 10.631706, 10.633798]
    # If this code passes the tests and is merged into `dev`,
    # the `new_loss` should be used as the benchmark for test cases.
    if args.use_new_loss:
        loss_std = new_loss_std
    else:
        loss_std = old_loss_std

    _base_train(args, loss_std, parallel_config)


def parallel_train_dp2(args):
    """test llama train in dp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    new_parallel_config = {
        'data_parallel': 2
    }

    loss_std = [10.624823, 10.626235, 10.635187, 10.623976, 10.627350,
                10.622116, 10.623417, 10.628310, 10.627144, 10.624737,
                10.622215, 10.610016, 10.618230, 10.605141, 10.615884,
                10.599818, 10.616759, 10.620478, 10.610780, 10.614862]

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


def parallel_train_mp2(args):
    """test llama train in mp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    new_parallel_config = {
        'tensor_parallel': 2
    }

    loss_std = [10.624356, 10.634629, 10.623917, 10.621689, 10.623317,
                10.607912, 10.608279, 10.621058, 10.618675, 10.612767,
                10.610808, 10.607621, 10.610729, 10.600761, 10.600204,
                10.586555, 10.591313, 10.595157, 10.594136, 10.600330]

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


def parallel_train_dp2_mp2(args):
    """test llama train in dp2mp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    new_parallel_config = {
        'tensor_parallel': 2
    }

    loss_std = [10.626953, 10.626993, 10.619582, 10.626608, 10.636086,
                10.626451, 10.625716, 10.638491, 10.616043, 10.610565,
                10.617392, 10.608209, 10.630942, 10.613029, 10.611053,
                10.607447, 10.618746, 10.607036, 10.602592, 10.620325]

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


TEST_MAP = {
    'single_train': single_train,
    'parallel_train_dp2': parallel_train_dp2,
    'parallel_train_mp2': parallel_train_mp2,
    'parallel_train_dp2_mp2': parallel_train_dp2_mp2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama model.')
    parser.add_argument('--experiment-mode', action='store_true', help='whether to use experiment mode.')
    parser.add_argument('--save-ckpt', action='store_true', help='whether to save checkpoint.')
    parser.add_argument('--load-ckpt', action='store_true', help='whether to load checkpoint.')
    parser.add_argument('--use-new-loss', action='store_true', help='whether to use new loss.')

    args_ = parser.parse_args()
    if args_.mode not in TEST_MAP:
        raise ValueError(f"Unsupported test mode: {args_.mode}")
    TEST_MAP[args_.mode](args_)
