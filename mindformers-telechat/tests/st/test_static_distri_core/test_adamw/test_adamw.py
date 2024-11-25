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
Test module for testing the AdamW interface used for MindFormers.
How to run this:
pytest tests/st/test_optim/test_adamw.py
"""
import numpy as np

from tests.st.test_static_distri_core.test_adamw.optimizer_util import (build_network, FakeNet,
                                                                        apex_fc1_weight_adamw_v,
                                                                        apex_fc2_weight_adamw_m,
                                                                        apex_fc2_weight_adamw_v,
                                                                        apex_fc1_weight_adamw_m)
from tests.utils.model_tester import ModelTester
from mindformers.core.context import build_context


def set_route_mode(mode):
    """set route mode."""
    runner = ModelTester(run_mode='train', batch_size=1)
    runner.args.mode = mode
    build_context(runner.args)



class TestAdamW:
    """A test class for testing optimizer computation."""

    def test_computation(self):
        """
        Feature: Trainer.train()
        Description: Test computation of AdamW in training.
        Expectation: AssertionError
        """
        set_route_mode(mode=0)

        config = {'type': 'AdamW', "weight_decay": 0.1}
        losses, cells = build_network(config, FakeNet(), is_group=True)

        print("cells.exp_avg[0].asnumpy():", cells.exp_avg[0].asnumpy())
        print("apex_fc1_weight_adamw_m:", apex_fc1_weight_adamw_m)
        print("cells.exp_avg_sq[0].asnumpy():", cells.exp_avg_sq[0].asnumpy())
        print("apex_fc1_weight_adamw_v:", apex_fc1_weight_adamw_v)

        print("cells.exp_avg[2].asnumpy():", cells.exp_avg[2].asnumpy())
        print("apex_fc2_weight_adamw_m:", apex_fc2_weight_adamw_m)
        print("cells.exp_avg_sq[2].asnumpy():", cells.exp_avg_sq[2].asnumpy())
        print("apex_fc2_weight_adamw_v:", apex_fc2_weight_adamw_v)

        print("losses:", losses)

        assert np.allclose(cells.exp_avg[0].asnumpy(), apex_fc1_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg[2].asnumpy(), apex_fc2_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[0].asnumpy(), apex_fc1_weight_adamw_v, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[2].asnumpy(), apex_fc2_weight_adamw_v, atol=1.e-4)
