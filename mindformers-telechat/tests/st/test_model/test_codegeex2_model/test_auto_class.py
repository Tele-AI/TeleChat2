# Copyright 2023 Huawei Technologies Co., Ltd
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
Test module for testing the codegeex2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_codegeex2_model/test_auto_class.py
"""
import os
import shutil
from mindspore import context

from mindformers import MindFormerBook, AutoModel, PreTrainedModel, AutoConfig, PretrainedConfig, AutoTokenizer, \
    PreTrainedTokenizerBase


class TestGLMAutoClassMethod:
    """A test class for testing Model classes"""

    def setup_method(self):
        """setup method."""
        context.set_context(mode=0)
        self.save_directory = MindFormerBook.get_default_checkpoint_save_folder()
        self.test_llm_list = ["codegeex2_6b"]

    def teardown_method(self):
        for model_or_config_type in self.test_llm_list:
            shutil.rmtree(os.path.join(self.save_directory, model_or_config_type), ignore_errors=True)

    def test_codegeex2_model(self):
        """
        Feature: AutoModel.
        Description: Test to get LL-Model instance by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        # input model name, load model and weights
        # Too time-cost, not used for now.
        for model_type in self.test_llm_list:
            model = AutoModel.from_pretrained(model_type, download_checkpoint=False)
            assert isinstance(model, PreTrainedModel)
            model.save_pretrained(
                save_directory=os.path.join(self.save_directory, model_type),
                save_name=model_type + "_model",
            )

    def test_codegeex2_config(self):
        """
        Feature: AutoConfig.
        Description: Test to get config instance by input config type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        # input model config name, load model and weights
        for config_type in self.test_llm_list:
            model_config = AutoConfig.from_pretrained(config_type)
            assert isinstance(model_config, PretrainedConfig)
            model_config.save_pretrained(
                save_directory=os.path.join(self.save_directory, config_type),
                save_name=config_type + "_config",
            )


    def test_codegeex2_tokenizer(self):
        """
        Feature: AutoTokenizer, input config.
        Description: Test to get tokenizer instance by input tokenizer type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        # input processor name
        tokenizer_type = "codegeex2_6b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer.save_pretrained(
            save_directory=os.path.join(self.save_directory, tokenizer_type),
            save_name=tokenizer_type + "_tokenizer",
        )
