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
Test Module for testing functions of AutoModel and BertForQuestionAnswering class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_qa_model\\test_qa_model.py
linux:  pytest ./tests/st/test_model/test_qa_model/test_qa_model.py
"""
import os

from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.models import BertForQuestionAnswering, PreTrainedModel
from mindformers.tools.logger import logger


class TestModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        # fine-tuning
        self.qa_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                              'qa')
        self.qa_config_path = os.path.join(MindFormerBook.get_project_path(),
                                           'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
        self.qa_config = AutoConfig.from_pretrained('qa_bert_base_uncased')

        # evaluation and prediction
        self.qa_squad_checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                    'qa')
        self.qa_squad_config_path = os.path.join(MindFormerBook.get_project_path(),
                                                 'configs', 'qa', 'run_qa_bert_base_uncased.yaml')
        self.qa_squad_config = AutoConfig.from_pretrained('qa_bert_base_uncased_squad')

        # save path
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(), 'qa')

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
        qa_model_a = AutoModel.from_pretrained('qa_bert_base_uncased', download_checkpoint=False)
        # input yaml path
        qa_model_c = AutoModel.from_config(self.qa_config_path, download_checkpoint=False)

        qa_model_a.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased')

        assert isinstance(qa_model_a, BertForQuestionAnswering)
        assert isinstance(qa_model_c, BertForQuestionAnswering)
        assert isinstance(qa_model_a, PreTrainedModel)
        assert isinstance(qa_model_c, PreTrainedModel)

        # evaluation and prediction test part
        # input model name
        qa_squad_model_a = AutoModel.from_pretrained('qa_bert_base_uncased_squad', download_checkpoint=False)
        # # input yaml path
        qa_squad_model_c = AutoModel.from_config(self.qa_squad_config_path, download_checkpoint=False)

        qa_squad_model_a.save_pretrained(self.save_directory, save_name='qa_bert_base_uncased_squad')

        assert isinstance(qa_squad_model_a, BertForQuestionAnswering)
        assert isinstance(qa_squad_model_c, BertForQuestionAnswering)
        assert isinstance(qa_squad_model_a, PreTrainedModel)
        assert isinstance(qa_squad_model_c, PreTrainedModel)
