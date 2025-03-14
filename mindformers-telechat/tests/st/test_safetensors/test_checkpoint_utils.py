#  Copyright 2024 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""test for load_checkpoint_utils."""
from unittest.mock import patch, MagicMock

import pytest

from mindformers.tools.register import MindFormerConfig
from mindformers.trainer.utils import build_model
from mindformers.utils.load_checkpoint_utils import CkptFormat, _get_checkpoint_mode, CheckpointFileMode, \
    _check_checkpoint_path

class TestCommonCheckpointMethod:
    """A test class for testing common methods"""
    def test_support_type(self):
        """test CkptFormat support type"""
        # run the test
        result = CkptFormat.support_type()

        # verify the results
        assert result == ['ckpt', 'safetensors']

    def test_check_checkpoint_path_with_non_string_pathlike(self):
        """test check checkpoint path with non string pathlike"""
        path = 123
        with pytest.raises(ValueError,
                           match=r"config.load_checkpoint must be a str, but got 123 as type <class 'int'>."):
            _check_checkpoint_path(path)

    def test_check_checkpoint_path_with_nonexistent_path(self):
        """test check checkpoint path with nonexistent path"""
        path = 'NoneExistPath'
        with pytest.raises(FileNotFoundError, match=r"config.load_checkpoint NoneExistPath does not exist."):
            _check_checkpoint_path(path)



class TestBuildModel:
    """A test class for testing build_model"""
    runner_config = {'sink_mode': True, 'epochs': 1, 'sink_size': 1}
    config = {'runner_config': runner_config}
    model = MagicMock()
    dataset = MagicMock()

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_sink_mode_value_error(self, mock_get_auto_parallel_context):
        """test build model sink mode value error"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        config.runner_config.sink_mode = False
        with pytest.raises(ValueError):
            build_model(config, None, None, False, False)

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_infer_predict_layout_when_do_eval_is_true(self, mock_get_auto_parallel_context):
        """test build model infer predict layout when do eval is true"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        build_model(config, self.model, self.dataset, True, False)
        self.model.infer_predict_layout.assert_called_once_with(*self.dataset)

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_infer_predict_layout_when_do_predict_is_true(self, mock_get_auto_parallel_context):
        """test build model infer predict layout when do predict is true"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        model = MagicMock()
        dataset = MagicMock()
        build_model(config, model, dataset, False, True)
        model.infer_predict_layout.assert_called_once_with(*dataset)

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_model_build(self, mock_get_auto_parallel_context):
        """test build model build"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        build_model(config, self.model, self.dataset, False, False)
        self.model.build.assert_called_once_with(train_dataset=self.dataset, epoch=1, sink_size=1)


class TestGetCheckpointMode:
    """A test class for testing get_checkpoint_mode"""
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_single_checkpoint_file(self, mock_isdir, mock_isfile):
        """test single checkpoint file"""
        mock_isfile.return_value = True
        mock_isdir.return_value = False
        config = type('', (), {})()
        config.load_checkpoint = '/test/checkpoint_file.safetensors'
        assert _get_checkpoint_mode(config) == CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_multi_checkpoint_file_with_rank_id(self, mock_isdir, mock_isfile):
        """test multi checkpoint file with rank id"""
        mock_isfile.return_value = False
        mock_isdir.return_value = True
        with patch('os.listdir', return_value=['rank_0']):
            config = type('', (), {})()
            config.load_checkpoint = '/test/checkpoint_dir/'
            assert _get_checkpoint_mode(config) == CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_multi_checkpoint_file(self, mock_isdir, mock_isfile):
        """ test multi checkpoint file"""
        mock_isfile.return_value = False
        mock_isdir.return_value = True
        with patch('os.listdir', return_value=['checkpoint.safetensors']):
            config = type('', (), {})()
            config.load_checkpoint = '/test/checkpoint_dir/'
            config.load_ckpt_format = '.safetensors'
            assert _get_checkpoint_mode(config) == CheckpointFileMode.MULTI_CHECKPOINT_FILE.value

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_invalid_path(self, mock_isdir, mock_isfile):
        """test invalid path"""
        mock_isfile.return_value = False
        mock_isdir.return_value = False
        config = type('', (), {})()
        config.load_checkpoint = 'invalid_path'
        with pytest.raises(ValueError, match="Provided path is neither a file nor a directory."):
            _get_checkpoint_mode(config)

    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_no_valid_checkpoint_files(self, mock_isdir, mock_isfile):
        """test no valid checkpoint files"""
        mock_isfile.return_value = False
        mock_isdir.return_value = True
        with patch('os.listdir', return_value=['not_a_checkpoint_file']):
            config = type('', (), {})()
            config.load_checkpoint = '/test/checkpoint_dir/'
            config.load_ckpt_format = '.safetensors'
            with pytest.raises(ValueError, match="not support mode: no valid checkpoint files found"):
                _get_checkpoint_mode(config)
