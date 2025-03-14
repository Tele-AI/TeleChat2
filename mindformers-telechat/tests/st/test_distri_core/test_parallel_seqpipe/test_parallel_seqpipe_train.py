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
""" Test SeqPipeline Parallel """
import os
import re
from multiprocessing.pool import Pool
import numpy as np
import pytest

os.environ['HCCL_BUFFSIZE'] = "200"
SH_PATH = os.path.split(os.path.realpath(__file__))[0]

def read_loss_from_log(file_path):
    """ reading loss from log """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            match_str = re.search(r'loss: (\d+\.\d+)', line)
            if match_str:
                loss_value = float(match_str.group(1))
                losses.append(loss_value)
    return losses

def run_command(command_info):
    """ run command """
    cmd, log_path = command_info
    print(f"\nrun cmd is:\n{cmd}")
    ret = os.system(cmd)
    return ret, log_path

def check_results(commands, results):
    """ check result (run successfully or not) """
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []

def compare_loss(pred_loss, gt_loss, model_name):
    """ compare given loss """
    pred_loss = np.array(pred_loss, np.float32)
    gt_loss = np.array(gt_loss, np.float32)

    print(f"seqpp loss: {pred_loss}", flush=True)
    print(f"golden loss: {gt_loss}", flush=True)
    assert np.allclose(pred_loss, gt_loss, atol=1e-3), f"{model_name} with seqpp " \
                                                            "loss accuracy test fail !"
    print("============== Interleaved staged pipeline net loss accuracy test pass !!! ==============")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestSeqPipelineParallel:
    """A test class for pipeline parallel interleaved. """

    @pytest.mark.skip(reason="Get golden loss from records")
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=1)
    def test_generate_golden_no_seqpipe(self):
        """
        Feature: generate pipeline net golden loss
        Description: run pynative mode pipeline net to generate golden loss
        Expectation: test success
        """
        scripts_name = "run_parallel_seqpipe.py"
        scripts_path = os.path.join(SH_PATH, scripts_name)

        llama_config_path = './finetune_llama3_8b_no_seqpp.yaml'
        llama_log_dir = "llama_no_seqpp"

        ds3_config_path = './pretrain_deepseek3_4layer_4p_no_seqpp.yaml'
        ds3_log_dir = "ds3_no_seqpp"
        ds3_register_path = os.path.abspath("../../../../research/deepseek3")

        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && msrun --worker_num=4 --local_worker_num=4 "
             f" --master_port=8120 --log_dir={llama_log_dir} --join=True --cluster_time_out=300 "
             f"{scripts_path} --config {llama_config_path}",
             f"{SH_PATH}/{llama_log_dir}/worker_0.log"),
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 && msrun --worker_num=4 --local_worker_num=4 "
             f" --master_port=8132 --log_dir={ds3_log_dir} --join=True --cluster_time_out=300 "
             f"{scripts_path} --config {ds3_config_path} --register_path {ds3_register_path}",
             f"{SH_PATH}/{ds3_log_dir}/worker_0.log")
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=2)
    def test_parallel_seqpipe_loss(self):
        """
        Feature: test pynative seqpipeline.
        Description: run pynative mode seqpipeline net to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_parallel_seqpipe.py"
        scripts_path = os.path.join(SH_PATH, scripts_name)

        llama_config_path = './finetune_llama3_8b.yaml'
        llama_log_dir = "llama_seqpp"

        ds3_config_path = './pretrain_deepseek3_4layer_4p.yaml'
        ds3_log_dir = "ds3_seqpp"
        ds3_register_path = os.path.abspath("../../../../research/deepseek3")

        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && msrun --worker_num=4 --local_worker_num=4 "
             f" --master_port=8120 --log_dir={llama_log_dir} --join=True --cluster_time_out=300 "
             f"{scripts_path} --config {llama_config_path}",
             f"{SH_PATH}/{llama_log_dir}/worker_0.log"),
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 && msrun --worker_num=4 --local_worker_num=4 "
             f" --master_port=8132 --log_dir={ds3_log_dir} --join=True --cluster_time_out=300 "
             f"{scripts_path} --config {ds3_config_path} --register_path {ds3_register_path}",
             f"{SH_PATH}/{ds3_log_dir}/worker_0.log")
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=3)
    def test_compare_loss_with_seqpp_or_not(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pipeline loss and golden loss which with shared weight
        Expectation: relative error smaller than 1e-3
        """
        llama_seqpp_log_path = './llama_seqpp/worker_3.log'
        llama_seqpp_loss = read_loss_from_log(llama_seqpp_log_path)
        llama_golden_loss = [11.782, 11.780, 11.766, 11.762, 11.757, 11.749, 11.747, 11.745]
        compare_loss(llama_seqpp_loss, llama_golden_loss, 'llama')

        ds3_seqpp_log_path = './ds3_seqpp/worker_3.log'
        ds3_seqpp_loss = read_loss_from_log(ds3_seqpp_log_path)
        ds3_golden_loss = [15.311, 15.275, 15.256, 15.242, 15.231, 15.221, 15.213, 15.208]
        compare_loss(ds3_seqpp_loss, ds3_golden_loss, 'deepseek3')
