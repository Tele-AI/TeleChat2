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
"""test parallel reducer on grads, is_finite and overflow status reduce"""

import argparse

import numpy as np
from mindspore import Parameter, Tensor
from mindspore.communication.management import init

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, GeneralConfig

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    initialize_model_parallel,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_data_parallel_rank,
)
from mindformers.experimental.parallel_core.pynative.training import ParallelTrainingReducer


def run_grads_reduce():
    """test grads reduce"""
    # TP 2 DP 2
    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=2,
        use_sequence_parallel=True,
    )
    training_config = GeneralConfig(parallel_config=parallel_config, loss_reduction="mean")

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
    )

    params = [
        Parameter(np.ones((3, 3)).astype(np.float32), name="norm"),
        Parameter(np.ones((3, 3)).astype(np.float32), name="weight"),
    ]
    parallel_reducer = ParallelTrainingReducer(params, training_config)

    grad_alpha_with_sp = (get_data_parallel_rank() * 2 + get_tensor_model_parallel_rank()) * 0.001
    grad_alpha = get_data_parallel_rank() * 2 * 0.001
    grads = [
        Tensor(np.ones((3, 3)).astype(np.float32) * grad_alpha_with_sp),
        Tensor(np.ones((3, 3)).astype(np.float32) * grad_alpha),
    ]
    parallel_reducer.inplace_reduce_grad(grads)

    golden_grad_alpha_with_sp_reduced = ((0.000 + 0.001) + (0.002 + 0.003)) / 2
    golden_grad_alpha_reduced = (0.000 + 0.002) / 2
    grads_golden = [
        Tensor(np.ones((3, 3)).astype(np.float32) * golden_grad_alpha_with_sp_reduced),
        Tensor(np.ones((3, 3)).astype(np.float32) * golden_grad_alpha_reduced),
    ]

    assert np.allclose(
        grads[0].asnumpy(), grads_golden[0].asnumpy(), 1.0e-5
    ), f"grads[0]: {grads[0]}, grads_golden[0]: {grads_golden[0]}"
    assert np.allclose(
        grads[1].asnumpy(), grads_golden[1].asnumpy(), 1.0e-5
    ), f"grads[1]: {grads[1]}, grads_golden[1]: {grads_golden[1]}"
    print("grads reduce test passed")


def run_overflow_reduce():
    """test overflow reduce"""
    # tp 2 pp 2
    parallel_config = ModelParallelConfig(tensor_model_parallel_size=2,\
                                          pipeline_model_parallel_size=2, micro_batch_num=1)
    training_config = GeneralConfig(parallel_config=parallel_config, loss_reduction="mean")

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
    )

    parallel_reducer = ParallelTrainingReducer([], training_config)

    # Test reduce over pp group
    if get_pipeline_model_parallel_rank() == 0:
        overflow_status = Tensor(True)
        is_finite = Tensor(True)
    else:
        overflow_status = Tensor(False)
        is_finite = Tensor(False)
    overflow_status = parallel_reducer.reduce_overflow(overflow_status)
    is_finite = parallel_reducer.reduce_is_finite(is_finite)

    assert overflow_status == Tensor(True), f"pp overflow_status reduce failed: {overflow_status}"
    assert is_finite == Tensor(False), f"pp is_finite reduce failed: {is_finite}"

    # Test reduce over tp group
    if get_tensor_model_parallel_rank() == 0:
        overflow_status = Tensor(True)
        is_finite = Tensor(True)
    else:
        overflow_status = Tensor(False)
        is_finite = Tensor(False)
    overflow_status = parallel_reducer.reduce_overflow(overflow_status)
    is_finite = parallel_reducer.reduce_is_finite(is_finite)

    assert overflow_status == Tensor(True), f"tp overflow_status reduce failed: {overflow_status}"
    assert is_finite == Tensor(False), f"tp is_finite reduce failed: {is_finite}"

    # Test reduce over tp pp group
    if get_tensor_model_parallel_rank() == 0 and get_pipeline_model_parallel_rank() == 0:
        overflow_status = Tensor(True)
        is_finite = Tensor(True)
    else:
        overflow_status = Tensor(False)
        is_finite = Tensor(False)

    overflow_status = parallel_reducer.reduce_overflow(overflow_status)
    is_finite = parallel_reducer.reduce_is_finite(is_finite)

    assert overflow_status == Tensor(True), f"tp pp overflow_status reduce failed: {overflow_status}"
    assert is_finite == Tensor(False), f"tp pp is_finite reduce failed: {is_finite}"

    print("overflow reduce test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad", help="Run grad reduce test.", action="store_true")

    args, rest_args = parser.parse_known_args()

    if args.grad:
        run_grads_reduce()
    else:
        run_overflow_reduce()
