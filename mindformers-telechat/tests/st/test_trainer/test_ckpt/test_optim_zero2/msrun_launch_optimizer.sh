#!/bin/bash
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

BASE_PATH=$(cd "(dirname $0)"; pwd)
USE_DEVICE_NUM=$1

msrun --worker_num=${USE_DEVICE_NUM} \
    --local_worker_num=${USE_DEVICE_NUM} \
    --master_port=8118 \
    --log_dir=msrun_log \
    --join=True \
    --cluster_time_out=300 \
    ${BASE_PATH}/run_optim_zero2.py
