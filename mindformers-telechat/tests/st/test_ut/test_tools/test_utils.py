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
"""test utils"""
from unittest import mock

import pytest

from mindformers.tools.utils import get_pipeline_rank_ids


class TestGetPipelineRankMethod:
    """A test class for testing tools utils method."""

    @pytest.mark.run(order=1)
    @mock.patch('mindformers.tools.utils.get_real_group_size')
    @mock.patch('mindformers.tools.utils.ms.get_auto_parallel_context')
    def test_get_pipeline_rank_ids_with_valid(self, mock_get_parallal_stage_num, mock_get_real_group_size):
        """test get pipeline rank ids in normal condition."""
        mock_get_real_group_size.return_value = 8
        mock_get_parallal_stage_num.return_value = 2

        test_rank_ids = get_pipeline_rank_ids()
        expected_ids = [0, 4]

        assert len(test_rank_ids) == len(expected_ids)
        assert test_rank_ids == expected_ids

    @pytest.mark.run(order=2)
    @mock.patch('mindformers.tools.utils.get_real_group_size')
    @mock.patch('mindformers.tools.utils.ms.get_auto_parallel_context')
    def test_get_pipeline_rank_ids_with_invalid(self, mock_get_parallal_stage_num, mock_get_real_group_size):
        """test get pipeline rank ids in normal condition."""
        mock_get_real_group_size.return_value = 8
        mock_get_parallal_stage_num.return_value = 3

        test_rank_ids = get_pipeline_rank_ids()

        expected_ids = [-1]

        assert len(test_rank_ids) == len(expected_ids)
        assert test_rank_ids == expected_ids
