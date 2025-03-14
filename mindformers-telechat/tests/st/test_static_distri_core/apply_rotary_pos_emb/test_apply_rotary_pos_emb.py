# pylint: skip-file
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
import os
import pytest

base_command = ('msrun --worker_num={device_num} --local_worker_num={device_num} '
                '--master_port=61371 --log_dir=msrun_log --join=True --cluster_time_out=300 '
                'run_apply_rotary_pos_emb.py --dp {dp} --cp {cp} --tp {tp}')


def build_msrun_command(device_num, dp, cp, tp):
    return base_command.format(device_num=device_num, dp=dp, cp=cp, tp=tp)


class TestApplyRotaryPosEmb:
    """A test class for testing SwiGlu"""

    @pytest.mark.level1
    @pytest.mark.env_onecard
    def test_apply_rotary_pos_emb_on_single(self):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test ApplyRotaryPosEmb
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        device_num = 1
        dp, cp, tp = (1, 1, 1)
        ret = os.system(build_msrun_command(device_num, dp, cp, tp))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0

    @pytest.mark.level1
    @pytest.mark.env_single
    def test_apply_rotary_pos_emb_on_parallel(self):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test ApplyRotaryPosEmb
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        device_num = 8
        dp, cp, tp = (2, 2, 2)
        ret = os.system(build_msrun_command(device_num, dp, cp, tp))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0

    @pytest.mark.level1
    @pytest.mark.env_single
    def test_apply_rotary_pos_emb_accuracy_dp(self):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test ApplyRotaryPosEmb accuracy dp
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        device_num = 2
        dp, cp, tp = (2, 1, 1)
        base_command = ('msrun --worker_num={device_num} --local_worker_num={device_num} '
                        '--master_port=61371 --log_dir=msrun_log --join=True --cluster_time_out=300 '
                        'run_apply_rotary_pos_emb_accuracy.py --dp {dp} --cp {cp} --tp {tp}')
        command = base_command.format(device_num=device_num, dp=dp, cp=cp, tp=tp)
        ret = os.system(command)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0

    @pytest.mark.level1
    @pytest.mark.env_single
    def test_apply_rotary_pos_emb_accuracy_tp(self):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test ApplyRotaryPosEmb accuracy tp
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        device_num = 2
        dp, cp, tp = (1, 1, 2)
        base_command = ('msrun --worker_num={device_num} --local_worker_num={device_num} '
                        '--master_port=61371 --log_dir=msrun_log --join=True --cluster_time_out=300 '
                        'run_apply_rotary_pos_emb_accuracy.py --dp {dp} --cp {cp} --tp {tp}')
        command = base_command.format(device_num=device_num, dp=dp, cp=cp, tp=tp)
        ret = os.system(command)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0
