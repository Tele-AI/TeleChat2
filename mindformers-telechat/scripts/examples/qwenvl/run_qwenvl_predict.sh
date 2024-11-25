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
TOKENIZER=$4
IMAGE_PATH=$5
PROMPT=$6
BATCH_SIZE=$7
DEVICE_NUM=$8

script_path="$(realpath "$(dirname "$0")")"

# set PYTHONPATH for research directory
export PYTHONPATH=$script_path/../../../:$script_path/../../../research/:$script_path/../../../research/qwenvl/:$PYTHONPATH

export GRAPH_OP_RUN=1

if [ "$PARALLEL" = "single" ]; then
  python "$script_path"/run_qwenvl_generate.py \
    --config "$CONFIG_PATH" \
    --load_checkpoint "$CKPT_PATH" \
    --vocab_file "$TOKENIZER" \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --batch_size $BATCH_SIZE \
    --use_past True \
    --predict_length 1024 \
    --seq_length 1024
elif [ "$PARALLEL" = "parallel" ]; then
  bash "$script_path"/../../msrun_launcher.sh \
    "$script_path/run_qwenvl_generate.py \
    --config $CONFIG_PATH \
    --load_checkpoint $CKPT_PATH \
    --vocab_file $TOKENIZER \
    --use_past False \
    --image_path '$IMAGE_PATH' \
    --prompt '$PROMPT' \
    --batch_size $BATCH_SIZE \
    --predict_length 1024 \
    --seq_length 1024 \
    --use_parallel True" "$DEVICE_NUM"
else
  echo "Only support 'single' or 'parallel', but got $PARALLEL."
fi
