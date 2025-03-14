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
Test Module for testing functions of AutoModel and BertForTokenClassification class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_tokcls_model\\test_tokcls_model.py
linux:  pytest ./tests/st/test_model/test_tokcls_model/test_tokcls_model.py
"""
import os
import shutil
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.models import BertForTokenClassification
from mindformers.tools.logger import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_class(self):
        """get_input"""
        # fine-tuning
        self.tokcls_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                  'tokcls')
        self.tokcls_config_path = os.path.join(MindFormerBook.get_project_path(),
                                               'configs', 'tokcls', 'run_tokcls_bert_base_chinese.yaml')
        self.tokcls_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese')

        # evaluation and prediction
        self.tokcls_cluener_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                          'tokcls')
        self.tokcls_cluener_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                       'configs', 'tokcls', 'run_tokcls_bert_base_chinese_cluener.yaml')
        self.tokcls_cluener_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'tokcls')

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

        # fine-tuning part
        # input model name
        tokcls_model_a = AutoModel.from_pretrained('tokcls_bert_base_chinese', download_checkpoint=False)

        tokcls_model_a.save_pretrained(self.save_directory, save_name='tokcls_bert_base_chinese')

        assert isinstance(tokcls_model_a, BertForTokenClassification)

        # evaluation and prediction test part
        # input model name
        tokcls_cluener_model_a = \
            AutoModel.from_pretrained('tokcls_bert_base_chinese_cluener', download_checkpoint=False)

        tokcls_cluener_model_a.save_pretrained(self.save_directory,
                                               save_name='tokcls_bert_base_chinese_cluener')

        assert isinstance(tokcls_cluener_model_a, BertForTokenClassification)
