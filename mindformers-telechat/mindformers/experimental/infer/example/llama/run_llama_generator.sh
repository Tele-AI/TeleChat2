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

# run parallel-llama generator
script_dir=$(dirname "$(realpath "$0")")

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config_path> <ckpt_path> <device_num> [<port>]"
    exit 1
fi

CONFIG_PATH=$1
CKPT_PATH=$2
DEVICE_NUM=$3

PORT=8124
if [ -n "$4" ]; then
    PORT=$4
fi

msrun --worker_num $DEVICE_NUM --local_worker_num $DEVICE_NUM --master_port $PORT --log_dir "output/msrun_log" --cluster_time_out 500 \
"$script_dir/run_llama_generator.py" --config_path $CONFIG_PATH --load_checkpoint $CKPT_PATH
