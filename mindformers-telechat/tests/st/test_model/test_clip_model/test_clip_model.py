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
Test Module for testing functions of AutoModel and CLIPModel class

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_clip_model\\test_clip_model.py
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_model.py
"""
import os
import time

import mindspore as ms

from mindformers import MindFormerBook, AutoConfig, AutoModel
from mindformers.models import CLIPModel, PreTrainedModel
from mindformers.tools.logger import logger

ms.set_context(mode=0)


class TestCLIPModelMethod:
    """A test class for testing Model classes"""
    def setup_method(self):
        """get_input"""
        self.model_type = "clip_vit_b_32"

        self.checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                           'clip')
        self.config_path = os.path.join(MindFormerBook.get_project_path(),
                                        'configs', 'clip', 'run_clip_vit_b_32_pretrain_flickr8k.yaml')
        self.config = AutoConfig.from_pretrained(self.model_type)

        self.checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                            'clip', self.model_type + '.ckpt')
        self.save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                           'clip')

    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        time.sleep(5)

        AutoModel.show_support_list()
        support_list = AutoModel.get_support_list()
        logger.info(support_list)
        # input model name, load model and weights
        model_a = AutoModel.from_pretrained(self.model_type)
        # input model directory, load model and weights
        model_b = AutoModel.from_pretrained(self.checkpoint_dir)
        # input yaml path, load model without weights
        model_c = AutoModel.from_config(self.config_path)
        # input config, load model without weights
        model_d = AutoModel.from_config(self.config)

        model_a.save_pretrained(self.save_directory, save_name=self.model_type)

        CLIPModel.show_support_list()
        support_list = CLIPModel.get_support_list()
        logger.info(support_list)
        # input model name, load model and weights
        model_i = CLIPModel.from_pretrained(self.model_type)
        # input model directory, load model and weights
        model_j = CLIPModel.from_pretrained(self.checkpoint_dir)
        # input config, load model weights
        model_k = CLIPModel(self.config)
        # input config, load model without weights
        self.config.checkpoint_name_or_path = None
        model_l = CLIPModel(self.config)

        model_i.save_pretrained(self.save_directory, save_name=self.model_type)

        # all models are ClipModel class， and inherited from PreTrainedModel
        assert isinstance(model_i, CLIPModel)
        assert isinstance(model_j, CLIPModel)
        assert isinstance(model_k, CLIPModel)
        assert isinstance(model_l, CLIPModel)

        assert isinstance(model_i, PreTrainedModel)
        assert isinstance(model_j, PreTrainedModel)
        assert isinstance(model_k, PreTrainedModel)
        assert isinstance(model_l, PreTrainedModel)

        # all models are CLIPModel class， and inherited from PreTrainedModel
        assert isinstance(model_a, CLIPModel)
        assert isinstance(model_b, CLIPModel)
        assert isinstance(model_c, CLIPModel)
        assert isinstance(model_d, CLIPModel)

        assert isinstance(model_a, PreTrainedModel)
        assert isinstance(model_b, PreTrainedModel)
        assert isinstance(model_c, PreTrainedModel)
        assert isinstance(model_d, PreTrainedModel)
