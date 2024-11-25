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

PARALLEL=$1
CONFIG_PATH=$2
CKPT_PATH=$3
VOCAB_FILE=$4
DEVICE_NUM=$5

script_path="$(realpath "$(dirname "$0")")"

if [ "$PARALLEL" = "single" ]; then
  python "$script_path"/run_llama3_generate.py \
    --config_path "$CONFIG_PATH" \
    --load_checkpoint "$CKPT_PATH" \
    --vocab_file "$VOCAB_FILE"
elif [ "$PARALLEL" = "parallel" ]; then
  bash "$script_path"/../../msrun_launcher.sh \
    "$script_path/run_llama3_generate.py \
    --config_path $CONFIG_PATH \
    --load_checkpoint $CKPT_PATH \
    --vocab_file $VOCAB_FILE \
    --use_parallel" "$DEVICE_NUM" 
else
  echo "Only support 'single' or 'parallel', but got $PARALLEL."
fi
