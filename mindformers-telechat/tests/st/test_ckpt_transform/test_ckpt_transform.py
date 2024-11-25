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
Test module for testing the TransformCkpt interface used for mindformers.
How to run this:
pytest tests/st/test_ckpt_transform/test_ckpt_transform.py
"""
import os
import shutil
import hashlib
import pytest

import mindspore as ms

from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Config

NUM_LAYERS = 2
HIDDEN_SIZE = 16
NUM_HEADS = 2
SEQ_LENGTH = 32
WORK_DIR = os.path.split(os.path.realpath(__file__))[0]
WEIGHT_PATH = os.path.join(WORK_DIR, "checkpoint_network.ckpt")
STRATEGY_8_NPUS = os.path.join(WORK_DIR, "strategy_8_npus.ckpt")
STRATEGY_4_NPUS = os.path.join(WORK_DIR, "strategy_4_npus.ckpt")
OUTPUT_1TO8_SINGLE = os.path.join(WORK_DIR, "ckpt_transform_1to8_single_process")
OUTPUT_1TO8_MULTI = os.path.join(WORK_DIR, "ckpt_transform_1to8_multi_process")
OUTPUT_8TO4_SINGLE = os.path.join(WORK_DIR, "ckpt_transform_8to4_single_process")
OUTPUT_8TO4_MULTI = os.path.join(WORK_DIR, "ckpt_transform_8to4_multi_process")
OUTPUT_4TO1_SINGLE = os.path.join(WORK_DIR, "ckpt_transform_4to1_single_process")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTransformCkpt:
    """A test class for testing pipeline."""

    @classmethod
    def setup_class(cls):
        """get a complete weight."""
        model_config = GPT2Config(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                                  num_heads=NUM_HEADS, seq_length=SEQ_LENGTH)
        model = GPT2LMHeadModel(model_config)
        ms.save_checkpoint(model.parameters_dict(), WEIGHT_PATH)

    @staticmethod
    def check_output(output, rank_size):
        """check transformed checkpoint and strategy exists under `output`."""
        for rank_id in range(rank_size):
            transformed_ckpt = os.path.join(output, \
                f"transformed_checkpoint/checkpoint_network/rank_{rank_id}/checkpoint_{rank_id}.ckpt")
            assert os.path.exists(transformed_ckpt), f"{transformed_ckpt} is not found!"

    @staticmethod
    def calculate_md5(file_path):
        """calculate the md5 value of file_path"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    @staticmethod
    def show_error(rank_size):
        """show error of mindformer log"""
        for rank_id in range(rank_size):
            os.system(f"grep -E 'ERROR|error' {WORK_DIR}/log/transform_{rank_id}.log -C 3")

    def compare_md5_of_transformed_checkpoint(self, output1, output2):
        """compare md5 value of transformed checkpoint"""
        if os.path.isfile(output1):
            transformed_ckpt_1 = output1
        else:
            transformed_ckpt_1 = os.path.join(output1, \
                f"transformed_checkpoint/checkpoint_network/rank_0/checkpoint_0.ckpt")
        md5_1 = self.calculate_md5(transformed_ckpt_1)

        if os.path.isfile(output2):
            transformed_ckpt_2 = output2
        else:
            transformed_ckpt_2 = os.path.join(output2, \
                f"transformed_checkpoint/checkpoint_network/rank_0/checkpoint_0.ckpt")
        md5_2 = self.calculate_md5(transformed_ckpt_2)
        assert md5_1 == md5_2, f"the md5 of transformed_ckpt_1 and transformed_ckpt_2 is not equal!\n\
            transformed_ckpt_1={transformed_ckpt_1}: md5={md5_1}\ntransformed_ckpt_2={transformed_ckpt_2}: md5={md5_2}"

    @pytest.mark.run(order=1)
    def test_auto_trans_ckpt_1to8_single_process(self):
        """
        Feature: Trainer.train()
        Description: Test transforming the complete weights to eight-card distributed weights.
        Expectation: AssertionError
        """
        src_ckpt = WEIGHT_PATH
        src_strategy = "None"
        dst_ckpt_dir = os.path.join(OUTPUT_1TO8_SINGLE, "transformed_checkpoint")
        dst_strategy = STRATEGY_8_NPUS
        os.system(f"python {WORK_DIR}/transform_checkpoint.py \
            --src_checkpoint {src_ckpt} --src_strategy {src_strategy} \
                --dst_checkpoint_dir {dst_ckpt_dir} --dst_strategy {dst_strategy}")
        self.check_output(OUTPUT_1TO8_SINGLE, 8)

    @pytest.mark.run(order=2)
    def test_handle_trans_ckpt_1to8_multi_process(self):
        """
        Feature: TransformCkpt.__call__()
        Description: Test transforming the complete weights to eight-card distributed weights,
        with `transform_process_num`=2.
        Expectation: AssertionError
        """
        src_ckpt = WEIGHT_PATH
        src_strategy = "None"
        dst_ckpt_dir = os.path.join(OUTPUT_1TO8_MULTI, "transformed_checkpoint")
        dst_strategy = STRATEGY_8_NPUS
        world_size = 8
        process_num = 2
        os.system(f"bash {WORK_DIR}/transform_checkpoint.sh \
            {src_ckpt} {src_strategy} {dst_ckpt_dir} {dst_strategy} {world_size} {process_num}")
        self.check_output(OUTPUT_1TO8_MULTI, 8)
        self.compare_md5_of_transformed_checkpoint(OUTPUT_1TO8_SINGLE, OUTPUT_1TO8_MULTI)

    @pytest.mark.run(order=3)
    def test_handle_trans_ckpt_8to4_single_process(self):
        """
        Feature: TransformCkpt.__call__()
        Description: Test transforming the eight-card distributed weights to four-card distributed weights.
        Expectation: AssertionError
        """
        src_ckpt = os.path.join(OUTPUT_1TO8_SINGLE, "transformed_checkpoint/checkpoint_network")
        src_strategy = STRATEGY_8_NPUS
        dst_ckpt_dir = os.path.join(OUTPUT_8TO4_SINGLE, "transformed_checkpoint")
        dst_strategy = STRATEGY_4_NPUS
        os.system(f"python {WORK_DIR}/transform_checkpoint.py \
            --src_checkpoint {src_ckpt} --src_strategy {src_strategy} \
                --dst_checkpoint_dir {dst_ckpt_dir} --dst_strategy {dst_strategy}")
        self.check_output(OUTPUT_8TO4_SINGLE, 4)

    @pytest.mark.run(order=4)
    def test_auto_trans_ckpt_8to4_multi_process(self):
        """
        Feature: Trainer.train()
        Description: Test transforming the eight-card distributed weights to four-card distributed weights,
        with `transform_process_num`=2.
        Expectation: AssertionError
        """
        src_ckpt = os.path.join(OUTPUT_1TO8_SINGLE, "transformed_checkpoint/checkpoint_network")
        src_strategy = STRATEGY_8_NPUS
        dst_ckpt_dir = os.path.join(OUTPUT_8TO4_MULTI, "transformed_checkpoint")
        dst_strategy = STRATEGY_4_NPUS
        world_size = 4
        process_num = 2
        os.system(f"bash {WORK_DIR}/transform_checkpoint.sh \
            {src_ckpt} {src_strategy} {dst_ckpt_dir} {dst_strategy} {world_size} {process_num}")
        self.check_output(OUTPUT_8TO4_MULTI, 4)
        self.compare_md5_of_transformed_checkpoint(OUTPUT_8TO4_SINGLE, OUTPUT_8TO4_MULTI)

    @pytest.mark.run(order=5)
    def test_handle_trans_ckpt_4to1(self):
        """
        Feature: TransformCkpt.__call__()
        Description: Test transforming the four-card distributed weights to complete weights.
        Expectation: AssertionError
        """
        src_ckpt = os.path.join(OUTPUT_8TO4_MULTI, "transformed_checkpoint/checkpoint_network")
        src_strategy = STRATEGY_4_NPUS
        dst_ckpt_dir = os.path.join(OUTPUT_4TO1_SINGLE, "transformed_checkpoint")
        dst_strategy = None
        os.system(f"python {WORK_DIR}/transform_checkpoint.py \
            --src_checkpoint {src_ckpt} --src_strategy {src_strategy} \
                --dst_checkpoint_dir {dst_ckpt_dir} --dst_strategy {dst_strategy}")
        self.check_output(OUTPUT_4TO1_SINGLE, 1)

        shutil.rmtree(OUTPUT_1TO8_SINGLE)
        shutil.rmtree(OUTPUT_1TO8_MULTI)
        shutil.rmtree(OUTPUT_8TO4_SINGLE)
        shutil.rmtree(OUTPUT_8TO4_MULTI)
        shutil.rmtree(OUTPUT_4TO1_SINGLE)
        os.remove(WEIGHT_PATH)
