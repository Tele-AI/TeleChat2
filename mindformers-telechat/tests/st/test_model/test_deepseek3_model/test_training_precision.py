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
"""
Test module for testing the deepseek3 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_deepseek3_model/test_training_precision.py
"""
import os
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers.modules.transformer.moe import MoEConfig

from tests.st.training_checker import TrainingChecker

for path in sys.path:
    if path.endswith('/testcases'):
        new_path = os.path.join(path, 'research')
        if new_path not in sys.path:
            sys.path.append(new_path)

    if path.endswith('/research'):
        new_path = os.path.join(path, 'deepseek3')
        if new_path not in sys.path:
            sys.path.append(new_path)

# pylint: disable=C0413
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_train import TrainingDeepseekV3ForCausalLM

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    step_num = 20
    batch_size = 4
    vocab_size = 152064
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestDeepseek3TrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        moe_config = MoEConfig(expert_num=16,
                               expert_group_size=8,
                               capacity_factor=1.5,
                               aux_loss_factor=0.05,
                               num_experts_chosen=4,
                               routing_policy="TopkRouterV2",
                               enable_sdrop=False,
                               balance_via_topk_bias=True,
                               topk_bias_update_rate=0.0001,
                               use_fused_ops_topkrouter=True,
                               group_wise_a2a=False,
                               shared_expert_num=1,
                               routed_scaling_factor=2.5,
                               norm_topk_prob=False,
                               first_k_dense_replace=1,
                               moe_intermediate_size=128,
                               topk_group=4,
                               n_group=8,
                               aux_loss_factors=[0.001, 0., 0.],
                               aux_loss_types=["expert", "device", "comm"],
                               z_loss_factor=0.0,
                               expert_model_parallel=1,
                               use_gating_sigmoid=True,)

        model_config = DeepseekV3Config(seq_length=512,
                                        hidden_size=1024,
                                        num_layers=3,
                                        vocab_size=152064,
                                        param_init_type="float32",
                                        mtp_depth=1,
                                        moe_config=moe_config,
                                        extend_method="None",
                                        return_extra_loss=True,
                                        init_method_std=0.006,)
        model = TrainingDeepseekV3ForCausalLM(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_ratio=0.01, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.95,
                                             eps=1.e-6,
                                             weight_decay=0.1,
                                             learning_rate=lr_schedule)

        loss_list_std = [15.534516, 15.543842, 15.535684, 15.540131,
                         15.544271, 15.543932, 15.5377655, 15.540722,
                         15.534813, 15.536912, 15.541159, 15.536707,
                         15.546314, 15.5291195, 15.53494, 15.542152,
                         15.5426445, 15.551411, 15.542769, 15.54012]
        callback = TrainingChecker(loss_list_std=loss_list_std)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset,
                                    callbacks=callback,
                                    optimizers=optimizer,)

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: AssertionError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.config.runner_config.sink_mode = False
        self.task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
        self.task_trainer.config.callbacks = self.task_trainer.config.callbacks[:1]
        self.task_trainer.train()
