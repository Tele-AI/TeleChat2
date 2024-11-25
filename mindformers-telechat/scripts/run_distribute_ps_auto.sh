#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "[INFO] Please set the environment variable: SERVER_ID, SERVER_NUM, MS_SCHED_HOST, MS_SCHED_PORT(optional)"
    echo "       SERVER_ID is the current server serial number. SERVER_NUM is the total server number."
    echo "       MS_SCHED_HOST is the schedule server ip. MS_SCHED_PORT is communication port."
    echo "       Such as: export SERVER_ID=0; export SERVER_NUM=1; export MS_SCHED_HOST=[HOST IP]; export MS_SCHED_PORT=[PORT]"
    echo "Usage Help: bash run_distribute_ps_auto.sh [CONFIG_PATH] [RUN_STATUS]"
    exit 1
fi
if [[ -z ${SERVER_ID} ]]; then
  echo "Please set environment variable: SERVER_ID"
  exit
fi
if [[ -z ${SERVER_NUM} ]]; then
  echo "Please set environment variable: SERVER_NUM"
  exit
fi
# Launch 8 workers in default.
device_nums=8
if [[ -z ${PER_DEVICE_NUMS} ]]; then
  echo "Environment variable: PER_DEVICE_NUMS is not set, use default value 8."
else
  echo "PER_DEVICE_NUMS is set, using ${PER_DEVICE_NUMS} devices per server."
  device_nums=${PER_DEVICE_NUMS}
fi
if [[ -z ${MS_SCHED_HOST} ]]; then
  echo "Please set environment variable: MS_SCHED_HOST"
  exit
fi
if [[ -z ${MS_SCHED_PORT} ]]; then
  export MS_SCHED_PORT=12420
  echo "Use default environment variable: MS_SCHED_PORT=${MS_SCHED_PORT}"
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# parse and check path
CONFIG_FILE=$(get_real_path $1)
RUN_STATUS=$2

# train command
ulimit -u unlimited
export MS_SERVER_NUM=0
export HCCL_BUFFSIZE=2048
host_ip=$(hostname -I | awk '{print $1}')
export HCCL_IF_IP=${host_ip}
export MS_SCHED_HOST=${MS_SCHED_HOST}  # Scheduler IP address
export MS_SCHED_PORT=${MS_SCHED_PORT}  # Scheduler port
export MS_ROLE=MS_WORKER
export MS_WORKER_NUM=$((${SERVER_NUM} * ${device_nums}))
export START_ID=$(($SERVER_ID * ${device_nums}))
export END_ID=$(($START_ID + ${device_nums}))

if [ -d "mindformers" ]; then
  cur_path=./
else
  cur_path=../
fi

rm -rf output
for((i=$START_ID;i<$END_ID;i++));
do
    export MS_NODE_ID=$i
    echo "[INFO] Start worker sched ip: ${MS_SCHED_HOST}, host ip: ${host_ip}, port: ${MS_SCHED_PORT}, " \
         "mode: MS_WORKER, work num: ${MS_WORKER_NUM}, start_id: ${START_ID}, end_id: ${END_ID}, node id: ${i}"
    rm -rf ./worker_$i
    mkdir ./worker_$i
    cp ${cur_path}/*.py ./worker_$i
    cp -r ${cur_path}/configs ./worker_$i
    cp -r ${cur_path}/mindformers ./worker_$i
    cd ./worker_$i || exit
    env > env.log
    python run_mindformer.py --config=$CONFIG_FILE --use_parallel=True --run_mode=$RUN_STATUS > work.log 2>&1 &
    cd ..
done

if [ $START_ID == 0 ]; then
  # Launch 1 scheduler.
  export MS_ROLE=MS_SCHED
  rm -rf ./sched
  mkdir ./sched
  cp ${cur_path}/*.py ./sched
  cp -r ${cur_path}/configs ./sched
  cp -r ${cur_path}/mindformers ./sched
  cd ./sched || exit
  echo "[INFO] Start scheduler sched ip: ${MS_SCHED_HOST}, host ip: ${host_ip}, port: ${MS_SCHED_PORT}, mode: MS_SCHED"
  python run_mindformer.py --config=$CONFIG_FILE --use_parallel=True --run_mode=$RUN_STATUS > sched.log 2>&1  &
  cd ..
fi

echo "[INFO] Startup completed. The log file is in worker_*/work.log or sched/sched.log in the current directory."
