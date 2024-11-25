# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Module for testing functions of AutoModel and BertForMultipleChoice class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_gpt2_model\\test_txtcls_model.py
linux:  pytest ./tests/st/test_model/test_gpt2_model/test_txtcls_model.py
"""
import os
import shutil
from mindformers import MindFormerBook, AutoModel, AutoConfig, GPT2ForSequenceClassification
from mindformers.tools.logger import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_class(self):
        """get_input"""
        # gpt2_txtcls
        self.gpt2_txtcls_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                       'gpt2')
        self.gpt2_txtcls_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                    'configs', 'gpt2', 'finetune_gpt2_small_txtcls_fp16.yaml')
        self.gpt2_txtcls_config = AutoConfig.from_pretrained('gpt2_txtcls')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'gpt2')

    def teardown_class(self):
        shutil.rmtree(self.save_directory, ignore_errors=True)

    # the first method to load model, AutoModel
    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()
        support_list = AutoModel.get_support_list()
        logger.info(support_list)

        # gpt2_txtcls
        gpt2_txtcls_model_a = AutoModel.from_pretrained('gpt2_txtcls', download_checkpoint=False)

        gpt2_txtcls_model_a.save_pretrained(self.save_directory, save_name='gpt2_txtcls')

        assert isinstance(gpt2_txtcls_model_a, GPT2ForSequenceClassification)
