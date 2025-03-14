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
WORKER_NUM=$1
LOCAL_WORKER=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NODE_RANK=$5

ROOT_PATH=`pwd`
bash ../../msrun_launcher.sh "run_mixtral.py \
    --config ${ROOT_PATH}/pretrain_mixtral-64x28b-8layers.yaml \
    --run_mode train \
    --use_parallel True" \
    $WORKER_NUM $LOCAL_WORKER $MASTER_ADDR $MASTER_PORT $NODE_RANK output/msrun False 300