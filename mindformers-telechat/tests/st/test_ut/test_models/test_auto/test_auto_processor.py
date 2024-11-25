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
""" test AutoProcessor."""
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
from pathlib import Path
import tempfile

from mindformers.models.auto import AutoProcessor
from mindformers import GPT2Processor

TEST_GPT2_NAME = "gpt2"
GPT2_REMOTE_PATH = "mindformersinfra/test_gpt2"


class TestAutoProcessor:
    """A test class for testing AutoProcessor."""

    def test_from_pretrained(self):
        """test AutoProcessor.from_pretrained() method with GPT2 model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "gpt2"
            save_dir = str(Path(tmp_dir) / "save")

            # experimental branch: load preprocessor.json from repo
            processor_exp = AutoProcessor.from_pretrained(GPT2_REMOTE_PATH, cache_dir=cache_dir)
            assert isinstance(processor_exp, GPT2Processor)
            processor_exp.save_pretrained(save_dir, save_json=True)

            # experimental branch: load preprocessor.json from local dir
            processor_exp_local = AutoProcessor.from_pretrained(save_dir)
            assert isinstance(processor_exp_local, GPT2Processor)

            # origin branch: load gpt2.yaml from this project
            processor_ori = AutoProcessor.from_pretrained(TEST_GPT2_NAME)
            processor_ori.save_pretrained(save_dir, save_json=False)

            # origin branch: load yaml when exists json file and yaml file at the same time
            assert os.path.exists(f"{save_dir}/tokenizer_config.json")
            assert os.path.exists(f"{save_dir}/mindspore_model.yaml")
