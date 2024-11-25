#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

if [ $# != 3 ]
then
  echo "Usage Help: bash run_standalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS] "
  exit 1
fi

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_FILE=$(check_real_path $1)
DEVICE_ID=$2
RUN_STATUS=$3

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

# get output path
output_dir=$(cat $CONFIG_FILE | grep output_dir)
output_dir=$(echo "$output_dir" | awk '{print $2}')
if [[ $output_dir =~ "'" ]]; then
  output_dir=${output_dir#*\'}
  output_dir=${output_dir%\'*}
else
  output_dir=${output_dir#*\"}
  output_dir=${output_dir%\"*}
fi
if [ ! -n "$output_dir" ]; then
  export LOCAL_DEFAULT_PATH="../../output"
  mkdir -p "../output"
  echo "output_dir is $(realpath "../output")"
elif [[ "$output_dir" =~ ^/ ]]; then
  export LOCAL_DEFAULT_PATH=$output_dir
  mkdir -p $LOCAL_DEFAULT_PATH
  echo "output_dir is $LOCAL_DEFAULT_PATH"
elif [[ "$output_dir" =~ ^\~ ]]; then
  home_path=`realpath ~`
  export LOCAL_DEFAULT_PATH="${home_path}${output_dir:1}"
  mkdir -p $LOCAL_DEFAULT_PATH
  echo "output_dir is $LOCAL_DEFAULT_PATH"
else
  export LOCAL_DEFAULT_PATH="../../$output_dir"
  mkdir -p "../$output_dir"
  echo "output_dir is $(realpath "../$output_dir")"
fi

# get log path
if [[ -z "$LOG_MF_PATH" ]]; then
  LOG_SAVE_PATH="../../output/log"
elif [[ "$LOG_MF_PATH" =~ ^\~ ]]; then
  home_path=`realpath ~`
  LOG_SAVE_PATH="${home_path}${LOG_MF_PATH:1}"
elif [[ ! "$LOG_MF_PATH" =~ ^/ ]]; then
  LOG_SAVE_PATH="../../$LOG_MF_PATH"
else
  LOG_SAVE_PATH=$LOG_MF_PATH
fi
export LOG_MF_PATH=$LOG_SAVE_PATH

ulimit -u unlimited
export DEVICE_ID=${DEVICE_ID}

rm -rf ./mf_standalone
mkdir ./mf_standalone
cp ../*.py ./mf_standalone
cp -r ../configs ./mf_standalone
cp -r ../mindformers ./mf_standalone
cd ./mf_standalone || exit
echo "start training for device $DEVICE_ID"
env > env.log
mkdir -p $LOG_MF_PATH/rank_0
python run_mindformer.py --config=$CONFIG_FILE --use_parallel=False --run_mode=$RUN_STATUS --device_id=$DEVICE_ID \
 --output_dir=$LOCAL_DEFAULT_PATH \
 &> $LOG_MF_PATH/rank_0/mindformer.log &
echo "log saved in $(realpath $LOG_MF_PATH)/rank_0"
cd ..

# if you want kill current job, you can use as follow:
# kill -9 $(ps aux | grep "python run_mindformer.py" | grep -v grep | awk '{print $2}')