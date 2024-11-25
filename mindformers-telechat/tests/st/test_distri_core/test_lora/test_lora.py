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
"""Test LoRA"""

import os
import pytest

import mindspore as ms


class TestLora:
    """A test class for testing lora."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "6,7"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_lora_pretrain(self):
        """
        Feature: test transformer block pretrain
        Description: run pynative mode to generate transformer block pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log_lora_pretrain " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd} " + \
              "--pretrain"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_pretrain/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_pretrain/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_lora_finetune(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log_lora_finetune " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_finetune/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_finetune/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_lora_checkpoint(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "lora_rank0_init.ckpt"
        lora_ckpt1_init = "lora_rank1_init.ckpt"
        lora_ckpt0 = "lora_rank0.ckpt"
        lora_ckpt1 = "lora_rank1.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.mapping',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()
