#!/bin/bash

# set PYTHONPATH
MF_SCRIPTS_ROOT=$(realpath "$(dirname "$0")")
export PYTHONPATH=$MF_SCRIPTS_ROOT/../../:$PYTHONPATH

USER_CMD="$*"
echo "Original command: $USER_CMD"

MODEL=""
MODEL_ARGS=""
TASKS=""
BATCH_SIZE=1
INCLUDE_PATH=""
REGISTER_PATH=""

function show_help {
    echo "  --register_path    <path>     Set the register path"
    echo "  --tasks            <tasks>    Set the tasks"
    echo "  --model_args       <args>     Set the model arguments"
    echo "  --model            <model>    Set the model"
    echo "  --batch_size       <size>     Set the batch size"
    echo "  --include_path     <path>     Set the include path"
    echo "  --help                        Show this help message"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --register_path) REGISTER_PATH="$2"; shift ;; # Set register path parameter
        --tasks) TASKS="$2"; shift ;;          # Set tasks parameter
        --model_args) MODEL_ARGS="$2"; shift ;; # Set model_args parameter
        --model) MODEL="$2"; shift ;;           # Set model parameter
        --batch_size) BATCH_SIZE="$2"; shift ;; # Set batch_size parameter
        --include_path) INCLUDE_PATH="$2"; shift ;; # Set include_path parameter
        --help)
          show_help
            return 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            return 1
            ;;
    esac
    shift
done

if [[ -n "$REGISTER_PATH" ]]; then
    export REGISTER_PATH=$REGISTER_PATH
    echo "REGISTER_PATH is: $REGISTER_PATH"
fi

MULTIPLE_CHOICE="cmmlu|ceval-valid|mmlu|race|lambada"
GENERATE_UNTIL="gsm8k|longbench|humaneval-x"

MULTIPLE_CHOICE_TASKS=0
GENERATE_UNTIL_TASKS=0

for TASK in $(echo "$TASKS" | tr ',' '\n'); do
    if [[ "$TASK" =~ $MULTIPLE_CHOICE ]]; then
        MULTIPLE_CHOICE_TASKS=1
    fi

    if [[ "$TASK" =~ $GENERATE_UNTIL ]]; then
        GENERATE_UNTIL_TASKS=1
    fi
done

if [[ $MULTIPLE_CHOICE_TASKS -eq 1 && $GENERATE_UNTIL_TASKS -eq 1 ]]; then
    echo "Error: Tasks cannot belong to both multiple_choice choice and generation_until tasks at the same time!"
    exit 1
fi

# multiple_choice choice
if [[ "$TASKS" =~ $MULTIPLE_CHOICE ]]; then
    MODEL_ARGS="$MODEL_ARGS,use_past=False"
    echo "Added 'use_past=False' to model_args for task '$TASKS'. Updated model_args: $MODEL_ARGS"
fi

# generation_until
if [[ "$TASKS" =~ $GENERATE_UNTIL ]]; then
    MODEL_ARGS="$MODEL_ARGS,use_past=True"
    echo "Added 'use_past=True' to model_args for task '$TASKS'. Updated model_args: $MODEL_ARGS"
fi

FINAL_CMD=""
if [[ -n "$INCLUDE_PATH" ]]; then
    FINAL_CMD="$FINAL_CMD --include_path $INCLUDE_PATH"
fi
if [[ "$MODEL_ARGS" == *"use_parallel=True"* ]]; then
    IFS=' ' read -r -a args <<< "$USER_CMD"
    last_option_index=-1
    for i in "${!args[@]}"; do
        if [[ ${args[$i]} == --* ]]; then
            last_option_index=$i
        fi
    done
    LAST_PARAMS="${args[*]:$((last_option_index + 2))}"
    FINAL_CMD="bash scripts/msrun_launcher.sh \"toolkit/benchmarks/eval_with_harness.py --model $MODEL --model_args $MODEL_ARGS --tasks $TASKS --batch_size $BATCH_SIZE $FINAL_CMD\" $LAST_PARAMS"
elif [[ "$MODEL_ARGS" != *"use_parallel=True"* ]]; then
    FINAL_CMD="python toolkit/benchmarks/eval_with_harness.py --model $MODEL --model_args $MODEL_ARGS --tasks $TASKS --batch_size $BATCH_SIZE $FINAL_CMD"
else
    echo "Error: Unknown command format."
    exit 1
fi

echo "Harness command: $FINAL_CMD"

eval "$FINAL_CMD"
