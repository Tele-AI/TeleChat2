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
"""logger for pipeline balance"""
import logging

DEFAULT_STDOUT_FORMAT = '%(levelname)s %(asctime)s %(filename)s:%(lineno)d - %(message)s'
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
OUTPUT_LEVEL_NUM = logging.WARNING
logging.addLevelName(OUTPUT_LEVEL_NUM, "OUTPUT")


def setup_logger(name: str, level: int = logging.DEBUG):
    """setup a logger"""
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(FORMATTER)

    def output(self, message, *args):
        self.warning(message, *args)

    logging.Logger.output = output
    ppb_logger = logging.getLogger(name)
    ppb_logger.setLevel(level)
    ppb_logger.addHandler(ch)

    return ppb_logger

logger = setup_logger('pipeline_balance', level=logging.INFO)
