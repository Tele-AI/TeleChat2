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
run_mode="single"
model_name_or_dir="predict_model"
predict_data="\"I love Beijing, because\" \"Huawei is a company that\""
device_num=2


# define help func
usage() {
  echo "Usage: bash $0 -m <mode> -n <model_name_or_dir> -i <predict_data> -d <device> -a <args>"
  exit 1
}

export TIME_RECORD='on'

# parsing parameters
OPTS=$(getopt -o m:n:i:d:a: --long mode:,model_name_or_dir:,predict_data:,device:,args: -- "$@")

if [ $? -ne 0 ]; then
  usage
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    --mode | -m )
      run_mode="$2"
      shift 2
      ;;
    --model_name_or_dir | -n )
      model_name_or_dir="$2"
      shift 2
      ;;
    --predict_data | -i )
      predict_data="$2"
      shift 2
      ;;
    --device | -d )
      device_num="$2"
      shift 2
      ;;
    --args | -a )
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


EXECUTION="$SCRIPT_PATH/run_inference.py \
 --model_name_or_dir $model_name_or_dir \
 --predict_data $predict_data \
 $script_args"
echo $EXECUTION

if [ "$run_mode" = "single" ]; then
  eval "python $EXECUTION"
elif [ "$run_mode" = "parallel" ]; then
  bash "$MF_ROOT_APTH"/scripts/msrun_launcher.sh "$EXECUTION" "$device_num"
else
  echo "Only support 'single' or 'parallel' mode, but got $PARALLEL."
fi
