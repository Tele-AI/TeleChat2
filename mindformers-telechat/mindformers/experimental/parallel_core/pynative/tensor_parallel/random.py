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
"""RNGStateTracer"""

__all__ = [
    'RNGStateTracer',
    'get_rng_tracer',
    'set_rng_seed',
    'DATA_PARALLEL_GENERATOR',
    'TENSOR_PARALLEL_GENERATOR',
    'EXPERT_PARALLEL_GENERATOR',
]

import random
from contextlib import contextmanager
import numpy as np
try:
    from mindspore import manual_seed, get_rng_state, set_rng_state
except ImportError:
    from mindspore.nn.generator import manual_seed, get_rng_state, set_rng_state
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_model_parallel_rank,
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
)


DATA_PARALLEL_GENERATOR = "dp_rng_generator"
TENSOR_PARALLEL_GENERATOR = "tp_rng_generator"
EXPERT_PARALLEL_GENERATOR = "exp_rng_generator"
IS_SEED_SET = False
CANDIDATE_MODES = [DATA_PARALLEL_GENERATOR, TENSOR_PARALLEL_GENERATOR, EXPERT_PARALLEL_GENERATOR]


class RNGStateTracer:
    """
    Examples:
        >>> with rngstatetracer.rng_fork():
        >>>     tensor = mint.normal(mean, std)
        >>> ...
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._states = {}

    def set_state(self, states):
        self._states = states

    def get_state(self):
        states = {}
        for mode in self._states:
            states[mode] = self._states[mode]
        return states

    def init_mode(self, mode, seed):
        "initialize mode and seed where mode should not be duplicate, otherwise reset should be called first"
        if mode in self._states:
            # if mode exists, raise exception
            raise ValueError(f"init generator with existed mode {mode}")
        # save current state, set and record target state, then restore old state
        orig_rng_state = get_rng_state()
        manual_seed(seed)
        self._states[mode] = get_rng_state()
        set_rng_state(orig_rng_state)

    # pylint: disable=W0101
    @contextmanager
    def rng_fork(self, mode=TENSOR_PARALLEL_GENERATOR):
        "fork rng state if seed is already set, otherwise keep the rng state unchanged"
        if not IS_SEED_SET:
            yield
            return
        # if mode not exists, raise exception
        if mode not in self._states:
            raise ValueError(f"not initialize or the parallel mode {mode} not exists ")
        # save current state, then set target state
        orig_rng_state = get_rng_state()
        set_rng_state(self._states[mode])
        try:
            # yield to do job
            yield
        finally:
            # restore old state
            self._states[mode] = get_rng_state()
            set_rng_state(orig_rng_state)

default_rng_tracer_ = None


def _init_default_rng_tracer():
    global default_rng_tracer_
    default_rng_tracer_ = RNGStateTracer()


def get_rng_tracer():
    if default_rng_tracer_ is None:
        _init_default_rng_tracer()
    return default_rng_tracer_


def parallel_mode_manual_seed(seed):
    "parallel_mode_manual_seed"
    dp_seed = seed
    tp_seed = seed + get_tensor_model_parallel_rank() + 2048
    exp_seed = seed + 1024 * 1 \
        + get_tensor_model_parallel_rank() + 2012

    tracer = get_rng_tracer()
    tracer.reset()
    manual_seed(seed) # for rng

    tracer.init_mode(DATA_PARALLEL_GENERATOR, dp_seed)
    tracer.init_mode(TENSOR_PARALLEL_GENERATOR, tp_seed)
    tracer.init_mode(EXPERT_PARALLEL_GENERATOR, exp_seed)


def set_rng_seed(seed, dp_random_init=False):
    seed = int(seed)
    seed = seed + 1024 * get_pipeline_model_parallel_rank()
    if dp_random_init:
        seed = seed + 64 * get_data_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    parallel_mode_manual_seed(seed) # for parallel rng tracer
    global IS_SEED_SET
    IS_SEED_SET = True
