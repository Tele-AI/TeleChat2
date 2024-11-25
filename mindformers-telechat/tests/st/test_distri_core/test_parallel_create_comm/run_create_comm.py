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
""" Test ParamAndGradBuffer """
import argparse
from mindspore.communication.management import init
from mindspore.communication import get_group_size, get_rank
# pylint: disable=W0401
from mindformers.experimental.parallel_core.pynative.parallel_state import *
import pytest

def run_initialize_and_destroy_model_parallel(order):
    """ run basic test for initialize and destroy comm groups """
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(order=order)
    init()
    assert not is_initialized()
    assert is_uninitialized()
    world_size = get_group_size()
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(tensor_model_parallel_size=2*world_size, order=order)
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(pipeline_model_parallel_size=2*world_size, order=order)
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(pipeline_model_parallel_size=world_size, \
                                         tensor_model_parallel_size=world_size, order=order)
    # Initialize
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, order=order)

    assert is_initialized()
    assert not is_uninitialized()
    assert model_parallel_is_initialized()

    assert get_tensor_model_parallel_group() is not None
    assert get_pipeline_model_parallel_group() is not None
    assert get_embedding_group is not None
    assert get_position_embedding_group is not None
    assert get_model_parallel_group() is not None
    assert get_data_parallel_group() is not None
    assert get_tensor_and_data_parallel_group() is not None
    # Destroy
    destroy_model_parallel()
    assert not is_initialized()
    assert is_uninitialized()
    with pytest.raises(RuntimeError):
        assert get_model_parallel_group()


def run_basic_world_size_rank_parallel_test(order):
    """ test world size and rank """
    init()
    world_size = get_group_size()
    rank = get_rank()

    # test tp
    initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert get_tensor_model_parallel_world_size() == world_size
    assert get_tensor_model_parallel_rank() == rank
    destroy_model_parallel()

    # test cp
    initialize_model_parallel(context_parallel_size=world_size, order=order)
    assert get_context_parallel_world_size() == world_size
    assert get_context_parallel_rank() == rank
    destroy_model_parallel()

    # test ep
    initialize_model_parallel(expert_model_parallel_size=world_size, order=order)
    assert get_data_modulo_expert_parallel_group() is not None
    assert get_data_parallel_world_size() == world_size
    assert get_data_parallel_rank() == rank
    assert get_expert_model_parallel_world_size() == world_size
    assert get_expert_model_parallel_rank() == rank
    assert get_tensor_and_expert_parallel_world_size() == world_size
    destroy_model_parallel()

    # test dp
    initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert get_data_parallel_world_size() == 1
    assert get_data_parallel_rank() == 0
    destroy_model_parallel()

    # test pp + vpp
    vpp_size = 4
    initialize_model_parallel(pipeline_model_parallel_size=world_size,
                              virtual_pipeline_model_parallel_size=vpp_size,
                              order=order)
    assert get_pipeline_model_parallel_world_size() == world_size
    assert get_pipeline_model_parallel_rank() == rank
    assert get_virtual_pipeline_model_parallel_world_size() == vpp_size
    set_virtual_pipeline_model_parallel_rank(0)
    assert is_pipeline_first_stage() == (rank == 0 and get_virtual_pipeline_model_parallel_rank() == 0)
    assert is_pipeline_first_stage(ignore_virtual=True) == (rank == 0)
    set_virtual_pipeline_model_parallel_rank(vpp_size-1)
    assert is_pipeline_last_stage() == (rank == world_size-1 and \
                                        get_virtual_pipeline_model_parallel_rank() == vpp_size-1)
    assert is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size-1)
    destroy_model_parallel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', required=True)

    args, rest_args = parser.parse_known_args()
    run_initialize_and_destroy_model_parallel(args.order)
    run_basic_world_size_rank_parallel_test(args.order)
