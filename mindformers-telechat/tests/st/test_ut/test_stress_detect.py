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
"""test stress detect."""
import logging
import unittest
from unittest.mock import patch
import pytest
from mindformers.core.callback import StressDetectCallBack
import mindspore as ms

ms.set_context(device_target='CPU')

PASS_CODE = 0
VOLTAGE_ERROR_CODE = 574007
OTHER_ERROR_CODE = 174003

logger = logging.getLogger('mindformers')

class TestStressDetectCallBack(unittest.TestCase):
    """A test class for testing StressDetectCallBack."""
    def setUp(self):
        self.detection_interval = 10
        self.num_detections = 1
        self.dataset_size = 1024

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.version_control.check_stress_detect_valid', return_value=True)
    def test_log_stress_detect_result_passed(self, _):
        """
        Feature: StressDetectCallBack
        Description: Test StressDetectCallBack log_stress_detect_result
        Expectation: No Exception
        """
        detect_ret_list = [PASS_CODE]

        callback = StressDetectCallBack(
            detection_interval=self.detection_interval,
            num_detections=self.num_detections,
            dataset_size=self.dataset_size
        )

        with self.assertLogs('mindformers', level='INFO') as log:
            callback.log_stress_detect_result(detect_ret_list)

        self.assertIn("INFO:mindformers:Stress detection passed", log.output)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.version_control.check_stress_detect_valid', return_value=True)
    def test_log_stress_detect_result_voltage_error(self, _):
        """
        Feature: StressDetectCallBack
        Description: Test StressDetectCallBack log_stress_detect_result
        Expectation: RuntimeError
        """
        detect_ret_list = [VOLTAGE_ERROR_CODE]

        callback = StressDetectCallBack(
            detection_interval=self.detection_interval,
            num_detections=self.num_detections,
            dataset_size=self.dataset_size
        )

        with self.assertRaises(RuntimeError) as context:
            callback.log_stress_detect_result(detect_ret_list)

        self.assertIn(f"Voltage recovery failed with error code: {VOLTAGE_ERROR_CODE}", str(context.exception))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.version_control.check_stress_detect_valid', return_value=True)
    def test_log_stress_detect_result_other_error(self, _):
        """
        Feature: StressDetectCallBack
        Description: Test StressDetectCallBack log_stress_detect_result
        Expectation: No Exception
        """
        detect_ret_list = [OTHER_ERROR_CODE]

        callback = StressDetectCallBack(
            detection_interval=self.detection_interval,
            num_detections=self.num_detections,
            dataset_size=self.dataset_size
        )

        with self.assertLogs('mindformers', level='WARNING') as log:
            callback.log_stress_detect_result(detect_ret_list)

        self.assertIn("WARNING:mindformers:Stress detection failed with error code: 174003", log.output)
