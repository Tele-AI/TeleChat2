# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023-2024 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Class Register Module For Pipeline."""
from typing import Any, Dict, List, Optional, Tuple, Union
from mindformers.tools.logger import logger


class PipelineRegistry:
    r"""Pipeline Registry.
    Args:
        supported_tasks (Dict[str, Any]): The task name supported.
        task_aliases (Dict[str, str]): The task alias mapping.
    """
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        """get the supported tasks"""
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        """check whether the task is in the supported_task list or not"""
        if task in self.task_aliases:
            task = self.task_aliases[task]
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return task, targeted_task, None

        if task.startswith("translation"):
            tokens = task.split("_")
            if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
                targeted_task = self.supported_tasks["translation"]
                task = "translation"
                return task, targeted_task, (tokens[1], tokens[3])
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
        )

    def register_pipeline(self,
                          task: str,
                          pipeline_class: type,
                          ms_model: Optional[Union[type, Tuple[type]]] = None,
                          default: Optional[Dict] = None,
                          task_type: Optional[str] = None) -> None:
        """Register custom pipeline objects"""
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}.")

        task_impl = {"impl": pipeline_class, "ms": ms_model}

        if default is not None:
            if "model" not in default and ("ms" in default):
                default = {"model": default}
            task_impl["default"] = default

        if task_type is not None:
            task_impl["type"] = task_type

        self.supported_tasks[task] = task_impl
        pipeline_class.registered_impl = {task: task_impl}

    def to_dict(self):
        return self.supported_tasks
