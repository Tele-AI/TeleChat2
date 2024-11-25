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

script_path="$(realpath "$(dirname "$0")")"

if [ "$PARALLEL" = "single" ]; then
  python "$script_path"/run_glm3_generate.py \
    --config_path "$CONFIG_PATH" \
    --load_checkpoint "$CKPT_PATH"
elif [ "$PARALLEL" = "multirole" ]; then
  python "$script_path"/run_glm3_generate.py \
    --config_path "$CONFIG_PATH" \
    --load_checkpoint "$CKPT_PATH" \
    --multi_role
else
  echo "Only support 'single' or 'multirole', but got $PARALLEL."
fi
