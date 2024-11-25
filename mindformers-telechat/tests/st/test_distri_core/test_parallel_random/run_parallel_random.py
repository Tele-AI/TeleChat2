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
"""run parallel linear"""
import time
import argparse
import numpy as np

import mindspore as ms
try:
    from mindspore import default_generator
except ImportError:
    from mindspore.nn.generator import default_generator
from mindspore import mint, Tensor, grad
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import (
    RNGStateTracer,
    get_rng_tracer,
    set_rng_seed,
    DATA_PARALLEL_GENERATOR,
    TENSOR_PARALLEL_GENERATOR,
)
from mindformers.experimental.parallel_core.pynative.dist_checkpointing import save_checkpoint, load_checkpoint

def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)

def run_random_tracer_parallel():
    """test ColumnParallelLinear."""
    tensor_parallel = 2
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    rank_id = get_rank()

    rng_tracer = get_rng_tracer()
    rng_tracer.reset()
    set_rng_seed(1000)

    mean = Tensor(generate_random_input((5, 5)))
    std = Tensor(generate_random_input((5, 5)))
    def step():
        return mint.normal(mean, std)

    result = []
    result.append(step())
    with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
        result.append(step())
    with get_rng_tracer().rng_fork(DATA_PARALLEL_GENERATOR):
        result.append(step())

    for mode in range(3):
        np.save(f"./result{mode}-rank{rank_id}.npy", np.array(result[mode].asnumpy()))
        print(f"result{mode} {result[mode]}")

    time.sleep(5)
    if rank_id != 0:
        return

    # save checkpoint
    append_dict = {}
    empty_cell = nn.Cell()
    append_dict = get_rng_tracer().get_state()
    append_dict["default_generator"] = default_generator
    ms.save_checkpoint(empty_cell, "gen.ckpt", append_dict=append_dict)
    # load into new tracer instance
    new_tracer = RNGStateTracer()
    param_dict = ms.load_checkpoint("gen.ckpt")
    target_state = {
        mode: param_dict[mode] for mode in [TENSOR_PARALLEL_GENERATOR, DATA_PARALLEL_GENERATOR] if mode in param_dict
    }
    new_tracer.set_state(target_state)

    # verify result
    world_size = get_group_size()
    for mode in range(3):
        candidate = [[] for _ in range(world_size)]
        for rank_id in range(world_size):
            candidate[rank_id] = np.load(f"./result{mode}-rank{rank_id}.npy", allow_pickle=True)
        if mode == 0: #in raw mode, the value should be the same in each rank
            assert np.allclose(candidate[0], candidate[1], rtol=1e-4, atol=1e-4)
            assert np.allclose(candidate[1], candidate[2], rtol=1e-4, atol=1e-4)
            assert np.allclose(candidate[2], candidate[3], rtol=1e-4, atol=1e-4)
        elif mode == 1: #in tp mode, the value should be the different in different tp rank
            assert np.allclose(candidate[0], candidate[2], rtol=1e-4, atol=1e-4)
            assert np.allclose(candidate[1], candidate[3], rtol=1e-4, atol=1e-4)
            assert not np.allclose(candidate[0], candidate[1], rtol=1e-4, atol=1e-4)
        else: #in default dp mode, the value should be the same in each rank
            assert np.allclose(candidate[0], candidate[1], rtol=1e-4, atol=1e-4)
            assert np.allclose(candidate[1], candidate[2], rtol=1e-4, atol=1e-4)
            assert np.allclose(candidate[2], candidate[3], rtol=1e-4, atol=1e-4)
    print("======= Test ckpt and rng =========")
    save_checkpoint(empty_cell)
    load_checkpoint("./", empty_cell)

def run_recompute_parallel():
    """run recompute with rng tracer in pynative parallel mode"""
    result = []
    class DropoutNet(nn.Cell):
        """tiny dropout net"""
        def __init__(self):
            super().__init__()
            self.dropout = mint.nn.Dropout(p=0.2)

        def construct(self, x):
            with get_rng_tracer().rng_fork():
                x = self.dropout(x)

            result.append(x.mean())
            print('forward results mean', result[len(result)-1])
            return x

    def forward(inputs, labels):
        logits = net(inputs)
        results = logits * labels
        return results, logits

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    init()
    initialize_model_parallel(tensor_model_parallel_size=2)
    set_rng_seed(1000)
    net = DropoutNet()
    net.set_train()

    net.recompute()

    inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
    labels = Tensor(np.random.randn(16, 10).astype(np.float32))

    grad_fn = grad(forward, grad_position=(0, 1), weights=None, has_aux=True)
    _, (_,) = grad_fn(inputs, labels)

    assert result[0] == result[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase', type=int, default=0, choices=[0, 1], help='choose testcase in [0, 1]')
    args = parser.parse_args()
    if args.testcase == 0:
        run_random_tracer_parallel()
    else:
        run_recompute_parallel()
