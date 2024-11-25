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
Test load safetensors.
How to run this:
pytest tests/st/test_model_runner/test_load_safetensors.py
"""
import os
import tempfile
import numpy as np
import pytest

import mindspore as ms

from mindformers import AutoModel, MindFormerConfig
from mindformers.model_runner import _transform_and_load_safetensors

ms.set_context(mode=0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestLoadSafetensors:
    """A test class for testing load safetensors."""
    def setup_method(self):
        """init model."""
        self.model = AutoModel.from_pretrained('llama2_7b', download_checkpoint=False, num_layers=1)

    @pytest.mark.run(order=1)
    def test_load(self):
        """
        Feature: _transform_and_load_safetensors()
        Description: Test load safetensors.
        """
        ms_model = ms.Model(self.model)
        input_ids = np.ones(shape=tuple([1, 2048]))
        inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as temp_file:
            temp_file_name = temp_file.name
            ms.save_checkpoint(self.model, temp_file_name,
                               choice_func=lambda x: x.startswith("lm_head"), format='safetensors')

            config = MindFormerConfig(**dict(load_checkpoint=None,
                                             load_safetensors=os.path.dirname(temp_file_name),
                                             output_dir=None,
                                             use_parallel=False,))

            _transform_and_load_safetensors(ms_model, self.model, inputs, config.load_checkpoint,
                                            config.load_safetensors, config.output_dir, config.use_parallel)
