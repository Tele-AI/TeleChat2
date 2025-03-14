#!/bin/bash
#
#  Copyright 2024 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
#

##
# Script Instruction
##

### Name:
### run_mindie.sh - Use to Start MindIE Service given a specific model
###
### Usage:
###   bash run_mindie.sh --model-name xxx --model-path /path/to/model
###
### Required:
###   --model-name             :Given a model name to identify MindIE Service.
###   --model-path             :Given a model path which contain necessary files such as yaml/conf.json/tokenizer/vocab etc.
### Options:
###   --help                   :Show this message.
###   --ip                     :The IP address bound to the MindIE Server business plane RESTful interface,default value: 127.0.0.1.
###   --port                   :The port bound to the MindIE Server business plane RESTful interface,default value: 1025.
###   --management-ip          :The IP address bound to the MindIE Server management plane RESTful interface,default value: 127.0.0.2.
###   --management-port        :The port bound to the MindIE Server management plane RESTful interface,default value: 1026.
###   --metrics-port           :The port bound to the performance indicator monitoring interface,default value: 1027.
###   --max-seq-len            :Maximum sequence length,default value: 2560.
###   --max-iter-times         :The global maximum output length of the model,default value: 512.
###   --max-input-token-len    :The maximum length of the token id,default value: 2048.
###   --max-prefill-tokens     :Each time prefill occurs, the total number of input tokens in the current batch,default value: 8192
###   --truncation             :Whether to perform parameter rationalization check interception,default value: false.
###   --template-type          :Reasoning type,default value: "Standard"
###   --max-preempt-count      :The upper limit of the maximum number of preemptible requests in each batch,default value: 0.
###   --support-select-batch   :Batch selection strategy,default value: false.
###   --npu-mem-size           :This can be used to apply for the upper limit of the KV Cache size in the NPU,default value: 8.
###   --max-prefill-batch-size :The maximum prefill batch size,default value: 50.
###   --world-size             :Enable several cards for inference.
###                             1. If it is not set, the parallel config in the YAML file is obtained by default. Set worldsize = dp*mp*pp.
###                             2. If set, modify the parallel config in the YAML file. set parallel config: dp:1 mp:worldSize pp:1
###   --ms-sched-host          :MS Scheduler IP address,default value: 127.0.0.1.
###   --ms-sched-port          :MS Scheduler port,default value: 8119.
###   For more details about config description, please check MindIE homepage: https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindiellm/llmdev/mindie_llm0004.html
help() {
  awk -F'### ' '/^###/ { print $2 }' "$0"
}

if [[ $# == 0 ]] || [[ "$1" == "--help" ]]; then
  help
  exit 1
fi

##
# Get device info
##
total_count=$(npu-smi info -l | grep "Total Count" | awk -F ':' '{print $2}' | xargs)

if [[ -z "$total_count" ]]; then
    echo "Error: Unable to retrieve device info. Please check if npu-smi is available for current user (id 1001), or if you are specifying an occupied device."
    exit 1
fi

echo "$total_count device(s) detected!"

##
# Set toolkit envs
##
echo "Setting toolkit envs..."
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]];then
                 source /usr/local/Ascend/ascend-toolkit/set_env.sh
         else
                 echo "ascend-toolkit package is incomplete please check it."
                 exit 1
         fi
echo "Toolkit envs set succeeded!"

##
# Set MindIE envs
##
echo "Setting MindIE envs..."
if [[ -f "/usr/local/Ascend/mindie/set_env.sh" ]];then
                 source /usr/local/Ascend/mindie/set_env.sh
         else
                 echo "mindie package is incomplete please check it."
                 exit 1
         fi
echo "MindIE envs set succeeded!"

##
# Default MS envs
##
ENV_MS_SCHED_HOST="127.0.0.1"
ENV_MS_SCHED_PORT="8119"

# Set PYTHONPATH
MF_SCRIPTS_ROOT=$(realpath "$(dirname "$0")")
export PYTHONPATH=$MF_SCRIPTS_ROOT/../:$PYTHONPATH

##
# Receive args and modify config.json
##
export MIES_INSTALL_PATH=/usr/local/Ascend/mindie/latest/mindie-service
CONFIG_FILE=${MIES_INSTALL_PATH}/conf/config.json
echo "MindIE Service config path:$CONFIG_FILE"
#default config
BACKEND_TYPE="ms"
MAX_SEQ_LEN=2560
MAX_PREFILL_TOKENS=8192
MAX_ITER_TIMES=512
MAX_INPUT_TOKEN_LEN=2048
TRUNCATION=false
HTTPS_ENABLED=false
MULTI_NODES_INFER_ENABLED=false
NPU_MEM_SIZE=8
MAX_PREFILL_BATCH_SIZE=50
TEMPLATE_TYPE="Standard"
MAX_PREEMPT_COUNT=0
SUPPORT_SELECT_BATCH=false
IP_ADDRESS="127.0.0.1"
PORT=1025
MANAGEMENT_IP_ADDRESS="127.0.0.2"
MANAGEMENT_PORT=1026
METRICS_PORT=1027

#modify config
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_WEIGHT_PATH="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift ;;
        --max-iter-times) MAX_ITER_TIMES="$2"; shift ;;
        --max-input-token-len) MAX_INPUT_TOKEN_LEN="$2"; shift ;;
        --max-prefill-tokens) MAX_PREFILL_TOKENS="$2"; shift ;;
        --truncation) TRUNCATION="$2"; shift ;;
        --world-size) WORLD_SIZE="$2"; shift ;;
        --template-type) TEMPLATE_TYPE="$2"; shift ;;
        --max-preempt-count) MAX_PREEMPT_COUNT="$2"; shift ;;
        --support-select-batch) SUPPORT_SELECT_BATCH="$2"; shift ;;
        --npu-mem-size) NPU_MEM_SIZE="$2"; shift ;;
        --max-prefill-batch-size) MAX_PREFILL_BATCH_SIZE="$2"; shift ;;
        --ip) IP_ADDRESS="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --management-ip) MANAGEMENT_IP_ADDRESS="$2"; shift ;;
        --management-port) MANAGEMENT_PORT="$2"; shift ;;
        --metrics-port) METRICS_PORT="$2"; shift ;;
        --ms-sched-host) ENV_MS_SCHED_HOST="$2"; shift ;;
        --ms-sched-port) ENV_MS_SCHED_PORT="$2"; shift ;;
        *)
            echo "Unknown parameter: $1"
            echo "Please check your inputs."
            exit 1
            ;;
    esac
    shift
done

##
# Set MS envs
##
echo "Setting MS envs..."
if [[ ! "$ENV_MS_SCHED_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "Error: MS_SCHED_HOST must be valid IP addresses. Current values: MS_SCHED_HOST=$ENV_MS_SCHED_HOST"
    exit 1
fi
if [[ ! "$ENV_MS_SCHED_PORT" =~ ^[0-9]+$ ]] || (( PORT < 1025 || PORT > 65535 )); then
    echo "Error: MS_SCHED_PORT must be integers between 1025 and 65535. Current values: MS_SCHED_PORT=$ENV_MS_SCHED_PORT"
    exit 1
fi
export MS_SCHED_HOST=$ENV_MS_SCHED_HOST
export MS_SCHED_PORT=$ENV_MS_SCHED_PORT
echo "MS envs set succeeded!"

if [ -z "$MODEL_WEIGHT_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: Both --model-path and --model-name are required."
    exit 1
fi
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_WEIGHT_PATH")}
echo "MODEL_NAME is set to: $MODEL_NAME"

#determine npu nums
EXT=".yaml"
YAML_FILE=$(find "$MODEL_WEIGHT_PATH" -type f -name "*$EXT")
if [[ -z "$YAML_FILE" ]]; then
    echo "Not found model .yaml file in given model path:$MODEL_WEIGHT_PATH"
    exit 1
fi
if [[ $YAML_FILE == *\ * ]]; then
    echo "Find multiple .yaml files in given model path:$MODEL_WEIGHT_PATH. Please keep only one yaml file needed."
    exit 1
fi
echo "model yaml is:$YAML_FILE"
USE_PARALLEL=$(cat $YAML_FILE | grep -w 'use_parallel' | awk '{print $2}'| tr -d '[:space:]')
if [[ -z "$WORLD_SIZE" ]]; then
  echo "worldSize is not set, obtain the parallel config from the yaml file."
  if [ "$USE_PARALLEL" = "True" ]; then
    DATA_PARALLEL=$(cat $YAML_FILE | grep -w 'data_parallel' | awk '{print $2}'| awk '{print $1 + 0}')
    MODEL_PARALLEL=$(cat $YAML_FILE | grep -w 'model_parallel' | awk '{print $2}'| awk '{print $1 + 0}')
    PIPELINE_STAGE=$(cat $YAML_FILE | grep -w 'pipeline_stage' | awk '{print $2}'| awk '{print $1 + 0}')
    WORLD_SIZE=$((DATA_PARALLEL * MODEL_PARALLEL * PIPELINE_STAGE))
    echo "USE_PARALLEL is True, WORLD_SIZE is product of data_parallel, model_parallel, and pipeline_stage: $WORLD_SIZE"
  else
    WORLD_SIZE=1
    echo "USE_PARALLEL is False, WORLD_SIZE is: $WORLD_SIZE"
  fi
else
  if ! [[ $WORLD_SIZE -gt 0 && $WORLD_SIZE -le $total_count ]];  then
    echo "worldSize must be a positive integer between 1 and $total_count."
    exit 1
  fi
  if [[ $WORLD_SIZE == 1 ]];  then
    sed -i "s/use_parallel: [a-zA-Z]*/use_parallel: False/" "$YAML_FILE"
  else
    sed -i "s/use_parallel: [a-zA-Z]*/use_parallel: True/" "$YAML_FILE"
  fi
  echo "worldSize is set to $WORLD_SIZE, modify parallel config in yaml file with:dp = 1/mp = $WORLD_SIZE/pp = 1."
  sed -i "s/data_parallel: [0-9]*/data_parallel: 1/" "$YAML_FILE"
  sed -i "s/model_parallel: [0-9]*/model_parallel: $WORLD_SIZE/" "$YAML_FILE"
  sed -i "s/pipeline_stage: [0-9]*/pipeline_stage: 1/" "$YAML_FILE"
fi
NPU_DEVICE_IDS=$(seq -s, 0 $(($WORLD_SIZE - 1)))

#validate config
if [[ "$BACKEND_TYPE" != "ms" ]]; then
    echo "Error: BACKEND must be 'ms'. Current value: $BACKEND_TYPE"
    exit 1
fi


if [[ ! "$IP_ADDRESS" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] ||
   [[ ! "$MANAGEMENT_IP_ADDRESS" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "Error: IP_ADDRESS and MANAGEMENT_IP_ADDRESS must be valid IP addresses. Current values: IP_ADDRESS=$IP_ADDRESS, MANAGEMENT_IP_ADDRESS=$MANAGEMENT_IP_ADDRESS"
    exit 1
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1025 || PORT > 65535 )) ||
   [[ ! "$MANAGEMENT_PORT" =~ ^[0-9]+$ ]] || (( MANAGEMENT_PORT < 1025 || MANAGEMENT_PORT > 65535 )); then
    echo "Error: PORT and MANAGEMENT_PORT must be integers between 1025 and 65535. Current values: PORT=$PORT, MANAGEMENT_PORT=$MANAGEMENT_PORT"
    exit 1
fi

if [ "$MAX_PREFILL_TOKENS" -lt "$MAX_SEQ_LEN" ]; then
    MAX_PREFILL_TOKENS=$MAX_SEQ_LEN
    echo "MAX_PREFILL_TOKENS was less than MAX_SEQ_LEN. Setting MAX_PREFILL_TOKENS to $MAX_SEQ_LEN"
fi

MODEL_CONFIG_FILE="${MODEL_WEIGHT_PATH}/config.json"
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: config.json file not found in $MODEL_WEIGHT_PATH."
    exit 1
fi
chmod 600 "$MODEL_CONFIG_FILE"
#update config file
chmod u+w ${MIES_INSTALL_PATH}/conf/
sed -i "s/\"backendType\"\s*:\s*\"[^\"]*\"/\"backendType\": \"$BACKEND_TYPE\"/" $CONFIG_FILE
sed -i "s/\"modelName\"\s*:\s*\"[^\"]*\"/\"modelName\": \"$MODEL_NAME\"/" $CONFIG_FILE
sed -i "s|\"modelWeightPath\"\s*:\s*\"[^\"]*\"|\"modelWeightPath\": \"$MODEL_WEIGHT_PATH\"|" $CONFIG_FILE
sed -i "s/\"maxSeqLen\"\s*:\s*[0-9]*/\"maxSeqLen\": $MAX_SEQ_LEN/" "$CONFIG_FILE"
sed -i "s/\"maxPrefillTokens\"\s*:\s*[0-9]*/\"maxPrefillTokens\": $MAX_PREFILL_TOKENS/" "$CONFIG_FILE"
sed -i "s/\"maxIterTimes\"\s*:\s*[0-9]*/\"maxIterTimes\": $MAX_ITER_TIMES/" "$CONFIG_FILE"
sed -i "s/\"maxInputTokenLen\"\s*:\s*[0-9]*/\"maxInputTokenLen\": $MAX_INPUT_TOKEN_LEN/" "$CONFIG_FILE"
sed -i "s/\"truncation\"\s*:\s*[a-z]*/\"truncation\": $TRUNCATION/" "$CONFIG_FILE"
sed -i "s|\(\"npuDeviceIds\"\s*:\s*\[\[\)[^]]*\(]]\)|\1$NPU_DEVICE_IDS\2|" "$CONFIG_FILE"
sed -i "s/\"worldSize\"\s*:\s*[0-9]*/\"worldSize\": $WORLD_SIZE/" "$CONFIG_FILE"
sed -i "s/\"httpsEnabled\"\s*:\s*[a-z]*/\"httpsEnabled\": $HTTPS_ENABLED/" "$CONFIG_FILE"
sed -i "s/\"templateType\"\s*:\s*\"[^\"]*\"/\"templateType\": \"$TEMPLATE_TYPE\"/" $CONFIG_FILE
sed -i "s/\"maxPreemptCount\"\s*:\s*[0-9]*/\"maxPreemptCount\": $MAX_PREEMPT_COUNT/" $CONFIG_FILE
sed -i "s/\"supportSelectBatch\"\s*:\s*[a-z]*/\"supportSelectBatch\": $SUPPORT_SELECT_BATCH/" $CONFIG_FILE
sed -i "s/\"multiNodesInferEnabled\"\s*:\s*[a-z]*/\"multiNodesInferEnabled\": $MULTI_NODES_INFER_ENABLED/" "$CONFIG_FILE"
sed -i "s/\"maxPrefillBatchSize\"\s*:\s*[0-9]*/\"maxPrefillBatchSize\": $MAX_PREFILL_BATCH_SIZE/" "$CONFIG_FILE"
sed -i "s/\"ipAddress\"\s*:\s*\"[^\"]*\"/\"ipAddress\": \"$IP_ADDRESS\"/" "$CONFIG_FILE"
sed -i "s/\"port\"\s*:\s*[0-9]*/\"port\": $PORT/" "$CONFIG_FILE"
sed -i "s/\"managementIpAddress\"\s*:\s*\"[^\"]*\"/\"managementIpAddress\": \"$MANAGEMENT_IP_ADDRESS\"/" "$CONFIG_FILE"
sed -i "s/\"managementPort\"\s*:\s*[0-9]*/\"managementPort\": $MANAGEMENT_PORT/" "$CONFIG_FILE"
sed -i "s/\"metricsPort\"\s*:\s*[0-9]*/\"metricsPort\": $METRICS_PORT/" $CONFIG_FILE
sed -i "s/\"npuMemSize\"\s*:\s*-*[0-9]*/\"npuMemSize\": $NPU_MEM_SIZE/" "$CONFIG_FILE"

##
# Start service
##
echo "Current configurations are displayed as follows:"
cat $CONFIG_FILE
npu-smi info -m > ~/device_info
nohup ${MIES_INSTALL_PATH}/bin/mindieservice_daemon > output.log 2>&1 &
echo "MindIE server has been started. You can exec command to check log: tail -f output.log"