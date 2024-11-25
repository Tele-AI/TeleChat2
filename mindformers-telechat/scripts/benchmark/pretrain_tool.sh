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

# set default value
model_name_or_dir="llama2_7b"
pretrain_data=""

# msrun Default Parameters
# The script defaults to a single process
WORKER_NUM=8
LOCAL_WORKER=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=8118
NODE_RANK=0
LOG_DIR="output/msrun_log"
JOIN="False"
CLUSTER_TIME_OUT=7200

# define help func
usage() {
  echo "Usage: bash $0 -n <model_name_or_dir> -i <pretrain_data> -w <worker_num> -l <local_worker> -a <master_addr> -p <master_port> -r <node_rank> -o <log_dir> -j <join> -t <cluster_timeout> --args <args>"
  exit 1
}

export TIME_RECORD='on'

# parsing parameters
OPTS=$(getopt -o n:i:w:l:a:p:r:o:j:t: --long model_name_or_dir:,pretrain_data:,worker_num:,local_worker:,master_addr:,master_port:,node_rank:,log_dir:,join:,cluster_timeout:,args: -- "$@")

if [ $? -ne 0 ]; then
  usage
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    --model_name_or_dir | -n )
      model_name_or_dir="$2"
      shift 2
      ;;
    --pretrain_data | -i )
      pretrain_data="$2"
      shift 2
      ;;
    --worker_num | -w )
      WORKER_NUM="$2"
      shift 2
      ;;
    --local_worker | -l )
      LOCAL_WORKER="$2"
      shift 2
      ;;
    --master_addr | -a )
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master_port | -p )
      MASTER_PORT="$2"
      shift 2
      ;;
    --node_rank | -r )
      NODE_RANK="$2"
      shift 2
      ;;
    --log_dir | -o )
      LOG_DIR="$2"
      shift 2
      ;;
    --join | -j )
      JOIN="$2"
      shift 2
      ;;
    --cluster_timeout | -t )
      CLUSTER_TIME_OUT="$2"
      shift 2
      ;;
    --args )
      script_args="$2"
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    * )
      usage
      ;;
  esac
done

# set environment
SCRIPT_PATH=$(realpath "$(dirname "$0")")
MF_ROOT_APTH=$(realpath "$SCRIPT_PATH/../../")
export PYTHONPATH=$MF_ROOT_APTH:$PYTHONPATH


EXECUTION="$SCRIPT_PATH/run_pretrain.py \
 --model_name_or_dir $model_name_or_dir \
 --pretrain_data $pretrain_data \
 $script_args"
echo $EXECUTION

# Integrate msrun_launcher.sh parameters directly into the script
if [ "$WORKER_NUM" -eq 1 ]; then
  echo "Running in single process mode."
  eval "python $EXECUTION"
else
  if [ "$WORKER_NUM" -eq "$LOCAL_WORKER" ]; then
    echo "Running on a single node with $WORKER_NUM devices."
    bash "$MF_ROOT_APTH"/scripts/msrun_launcher.sh "$EXECUTION" "$WORKER_NUM"
  else
    bash "$MF_ROOT_APTH"/scripts/msrun_launcher.sh \
    "$EXECUTION" \
    "$WORKER_NUM" \
    "$LOCAL_WORKER" \
    "$MASTER_ADDR" \
    "$MASTER_PORT" \
    "$NODE_RANK" \
    "$LOG_DIR" \
    "$JOIN" \
    "$CLUSTER_TIME_OUT"

    echo "Running on multiple nodes."
  fi
fi