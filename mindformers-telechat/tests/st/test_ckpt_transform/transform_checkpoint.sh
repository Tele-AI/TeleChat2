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
script_dir=$(dirname "$(realpath "$0")")

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <src_ckpt> <src_strategy> <dst_ckpt_dir> <dst_strategy> <world_size> <process_num> [<prefix>]"
    exit 1
fi

SRC_CKPT=$(check_real_path $1)
SRC_STRATEGY="$2"
DST_CKPT_DIR=$(check_real_path $3)
DST_STRATEGY=$(check_real_path $4)
WORLD_SIZE=$5
PROCESS_NUM=$6
PREFIX="checkpoint_"
if [ -n "$7" ]; then
    PREFIX=$7
fi

# If src_strategy is "None", set it to an empty string
if [ "$SRC_STRATEGY" == "None" ]; then
    SRC_STRATEGY=""
else
    SRC_STRATEGY=$(check_real_path $SRC_STRATEGY)
fi

rm -rf ./log
mkdir -p ./log

for((i=0;i<${PROCESS_NUM};i++))
do
    rank_id=$((i * (WORLD_SIZE / PROCESS_NUM)))
    echo "start device_$i with rank_id $rank_id"
    export RANK_ID=$rank_id
    python $script_dir/transform_checkpoint.py \
    --src_checkpoint=$SRC_CKPT \
    --dst_checkpoint_dir=$DST_CKPT_DIR \
    --src_strategy=$SRC_STRATEGY \
    --dst_strategy=$DST_STRATEGY \
    --prefix=$PREFIX \
    --rank_id=$rank_id \
    --world_size=$WORLD_SIZE \
    --transform_process_num=$PROCESS_NUM \
    &> ./log/transform_$i.log 2>&1 &
    echo "transform log saved in $(realpath $PWD)/log/transform_$i.log"
done

wait

echo "transform finished"
