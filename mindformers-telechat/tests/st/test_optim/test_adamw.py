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
import pytest
import numpy as np

import mindspore as ms

from tests.st.test_optim.optimizer_util import build_network, FakeNet, default_fc1_weight_adamw_m, \
    default_fc2_weight_adamw_m, default_fc1_weight_adamw_v, default_fc2_weight_adamw_v

ms.set_context(mode=0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestAdamW:
    """A test class for testing optimizer computation."""
    def test_computation(self):
        """
        Feature: Trainer.train()
        Description: Test computation of AdamW in training.
        Expectation: AssertionError
        """
        config = {'type': 'AdamW', "weight_decay": 0.1}
        _, cells = build_network(config, FakeNet(), is_group=True)
        assert np.allclose(cells.exp_avg[0].asnumpy(), default_fc1_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg[2].asnumpy(), default_fc2_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[0].asnumpy(), default_fc1_weight_adamw_v, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[2].asnumpy(), default_fc2_weight_adamw_v, atol=1.e-4)
