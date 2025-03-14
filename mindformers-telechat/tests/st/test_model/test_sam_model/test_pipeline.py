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
Test module for testing the llama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llama_model/test_pipeline.py
"""
import cv2
import numpy as np

import mindspore as ms

from mindformers import pipeline

ms.set_context(mode=0)


class TestSamPipelineMethod:
    """A test class for testing pipeline."""
    def setup_method(self):
        """setup method."""
        self.test_llm_list = ['sam_vit_b']

    def test_pipeline(self):
        """
        Feature: pipeline.
        Description: Test pipeline by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        for model_type in self.test_llm_list:
            task_pipeline = pipeline(task='segment_anything', model=model_type)

            image = cv2.imread("scripts/examples/segment_anything/images/truck.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 0.使用task_pipeline.set_image提前抽取图像特征
            task_pipeline.set_image(image)

            # 1. 单点确定一个物体
            input_point = np.array([[500, 375]])
            input_label = np.array([1])
            task_pipeline({"points": input_point,
                           "labels": input_label},
                          multimask_output=True)

            # 2.两点确定相同物体
            input_point = np.array([[500, 375], [1125, 625]])
            input_label = np.array([1, 1])
            task_pipeline({"points": input_point,
                           "labels": input_label},
                          multimask_output=False)

            # 3.两点确定不同物体
            input_point = np.array([[[500, 375]],
                                    [[1125, 625]]])
            input_label = np.array([[1], [1]])
            task_pipeline({"points": input_point,
                           "labels": input_label},
                          multimask_output=False)

            # 4.一个前景点和背景点
            input_point = np.array([[500, 375], [1125, 625]])
            input_label = np.array([1, 0])
            task_pipeline({"points": input_point,
                           "labels": input_label},
                          multimask_output=False)

            # 5.单框确定一个物体
            input_box = np.array([425, 600, 700, 875])
            task_pipeline({"boxes": input_box},
                          multimask_output=False)

            # 6.框和背景点确定物体
            input_box = np.array([425, 600, 700, 875])
            input_point = np.array([[575, 750]])
            input_label = np.array([0])
            task_pipeline({"points": input_point,
                           "labels": input_label,
                           "boxes": input_box},
                          multimask_output=False)

            # 7.多组框和点确定不同物体
            input_boxes = np.array([[425, 600, 700, 875],
                                    [1360, 525, 1680, 780]])
            input_points = np.array([[[575, 750]],
                                     [[1525, 670]]])
            input_labels = np.array([[1], [1]])
            task_pipeline({"points": input_points,
                           "labels": input_labels,
                           "boxes": input_boxes},
                          multimask_output=False)

            # 8.多个框确定不同物体
            input_boxes = np.array([[75, 275, 1725, 850],
                                    [425, 600, 700, 875],
                                    [1375, 550, 1650, 800],
                                    [1240, 675, 1400, 750]])
            task_pipeline({"boxes": input_boxes},
                          multimask_output=False)

            # 单点确定一个物体: 传入cv2图像和prompt
            input_point = np.array([[500, 375]])
            input_label = np.array([1])

            image = cv2.imread("scripts/examples/segment_anything/images/truck.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            task_pipeline({"image": image,
                           "points": input_point,
                           "labels": input_label},
                          multimask_output=True)

            # 单点确定一个物体: 传入图像路径和prompt
            task_pipeline({"image": "scripts/examples/segment_anything/images/truck.jpg",
                           "points": input_point,
                           "labels": input_label},
                          multimask_output=True)
