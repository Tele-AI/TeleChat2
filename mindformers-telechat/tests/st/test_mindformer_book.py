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
Test Module for testing functions of MindFormerBook class

How to run this:
windows:  pytest .\\tests\\st\\test_mindformer_book.py
linux:  pytest ./tests/st/test_mindformer_book.py
"""
import pytest
from mindformers import MindFormerBook
from mindformers.tools.logger import logger


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindformer_book_show():
    """
    Feature: mindformers book class, show content
    Description: Test to show content in MindFormerBook
    Expectation: success
    """
    MindFormerBook.show_trainer_support_task_list()
    MindFormerBook.show_pipeline_support_task_list()
    MindFormerBook.show_model_support_list()
    MindFormerBook.show_model_ckpt_url_list()
    MindFormerBook.show_model_config_url_list()
    MindFormerBook.show_project_path()
    MindFormerBook.show_default_checkpoint_download_folder()
    MindFormerBook.show_default_checkpoint_save_folder()
    MindFormerBook.show_model_config_to_name()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindformer_book_get():
    """
    Feature: mindformers book class, get content
    Description: Test to get content of MindFormerBook
    Expectation: success
    """
    logger.info(MindFormerBook.get_trainer_support_task_list())
    logger.info(MindFormerBook.get_pipeline_support_task_list())
    logger.info(MindFormerBook.get_model_support_list())
    logger.info(MindFormerBook.get_model_ckpt_url_list())
    logger.info(MindFormerBook.get_model_config_url_list())
    logger.info(MindFormerBook.get_project_path())
    logger.info(MindFormerBook.get_default_checkpoint_download_folder())
    logger.info(MindFormerBook.get_default_checkpoint_save_folder())
    logger.info(MindFormerBook.get_model_config_to_name())


# default path is settable
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindformer_book_set():
    """
    Feature: mindformers book class, get content
    Description: Test to set content for MindFormerBook
    Expectation: success or TypeError
    """
    path = MindFormerBook.get_project_path()
    MindFormerBook.set_default_checkpoint_download_folder(path)
    MindFormerBook.set_default_checkpoint_save_folder(path)
    logger.info(MindFormerBook.get_default_checkpoint_download_folder())
    logger.info(MindFormerBook.get_default_checkpoint_save_folder())

    MindFormerBook.set_model_config_to_name("ModelConfig", "ModelName")
