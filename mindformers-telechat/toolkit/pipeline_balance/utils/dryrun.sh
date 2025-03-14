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
export GLOG_v=2
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3
export MS_SIMULATION_LEVEL=1
export MS_MEMORY_STATISTIC=1

if [ $# != 4 ]
then
  echo "Usage Help: bash dry_run.sh [EXECUTE_ORDER] [RANK_SIZE] [PIPELINE_STAGES]"
  exit 1
fi

EXECUTE_ORDER=$1
RANK_SIZE=$2
PIPELINE_STAGES=$3
OUTPUT_FOLDER=$4
RANK_GAP=$((RANK_SIZE/PIPELINE_STAGES)) 

export RANK_SIZE=$RANK_SIZE
unset RANK_TABLE_FILE

shopt -s extglob

for((i=0; i<$PIPELINE_STAGES; i++))
do
    export STAGE_ID=$i
    export RANK_ID=$(((i)*RANK_GAP))
    mkdir -p ./$OUTPUT_FOLDER/log/stage${STAGE_ID}_$RANK_ID
    echo "start training for rank $RANK_ID, stage $STAGE_ID"
    $EXECUTE_ORDER &> ./$OUTPUT_FOLDER/log/stage${STAGE_ID}_$RANK_ID/mindformer.log &
done

shopt -u extglob
