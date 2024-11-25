# 高阶接口开发样例

MindFormers套件的Trainer高阶接口提供了`train`、`finetune`、`evaluate`、`predict`
4个关键属性函数，帮助用户快速拉起任务的训练、微调、评估、推理流程：[Trainer.train](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L334) [Trainer.finetune](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L419) [Trainer.evaluate](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L516) [Trainer.predict](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L583)

使用`Trainer.train` `Trainer.finetune` `Trainer.evaluate` `Trainer.predict` 拉起任务的训练、微调、评估、推理流程，以下为使用`Trainer`
高阶接口进行全流程开发的使用样例。

## 单卡使用样例

- 样例代码`standalone_task.py`

```python
import argparse
import mindspore as ms

from mindformers import Trainer, TrainingArguments


def main(run_mode='train',
         task='text_generation',
         model_type='gpt2',
         pet_method='',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='hello!'):
    # 环境初始化
    ms.set_context(mode=0, device_target="Ascend", device_id=0)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=2, learning_rate=0.001, warmup_steps=100,
                                      sink_mode=True, sink_size=2)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task,
                   model=model_type,
                   pet_method=pet_method,
                   args=training_args,
                   train_dataset=train_dataset,
                   eval_dataset=eval_dataset)
    if run_mode == 'train':
        task.train()
    elif run_mode == 'finetune':
        task.finetune()
    elif run_mode == 'eval':
        task.evaluate()
    elif run_mode == 'predict':
        result = task.predict(input_data=predict_data)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='train', required=True, help='set run mode for model.')
    parser.add_argument('--task', default='text_generation', required=True, help='set task type.')
    parser.add_argument('--model_type', default='gpt2', required=True, help='set model type.')
    parser.add_argument('--train_dataset', default=None, help='set train dataset.')
    parser.add_argument('--eval_dataset', default=None, help='set eval dataset.')
    parser.add_argument('--predict_data', default='hello!', help='input data used to predict.')
    parser.add_argument('--pet_method', default='', help="set finetune method, now support type: ['', 'lora']")
    args = parser.parse_args()
    main(run_mode=args.run_mode,
         task=args.task,
         model_type=args.model_type,
         pet_method=args.pet_method,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data)
```

- 启动任务：

```shell
# 训练
python standalone_task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode train

# 评估
python standalone_task.py --task text_generation --model_type gpt2 --eval_dataset ./eval --run_mode eval

# 微调，支持
python standalone_task.py --task text_generation --model_type gpt2 --train_dataset ./finetune --pet_method lora --run_mode finetune

# 推理
python standalone_task.py --task text_generation --model_type gpt2 --predict_data 'hello!' --run_mode predict
```

## 分布式多卡使用样例

- 样例代码`distribute_task.py`

```python
import argparse

from mindformers import Trainer, TrainingArguments
from mindformers import init_context, ContextConfig, ParallelContextConfig


def context_init(optimizer_parallel=False):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                            gradients_mean=False,
                                            enable_parallel_optimizer=optimizer_parallel,
                                            full_batch=True)
    rank_id, device_num = init_context(use_parallel=True,
                                       context_config=context_config,
                                       parallel_config=parallel_config)


def main(run_mode='train',
         task='text_generation',
         model_type='gpt2',
         pet_method='',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='hello!',
         dp=1, mp=1, pp=1, micro_size=1, op=False):
    # 环境初始化
    context_init(optimizer_parallel=op)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=2, learning_rate=0.001, warmup_steps=100,
                                      sink_mode=True, sink_size=2)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task,
                   model=model_type,
                   pet_method=pet_method,
                   args=training_args,
                   train_dataset=train_dataset,
                   eval_dataset=eval_dataset)
    task.set_parallel_config(data_parallel=dp,
                             model_parallel=mp,
                             pipeline_stage=pp,
                             micro_batch_num=micro_size)
    if run_mode == 'train':
        task.train()
    elif run_mode == 'finetune':
        task.finetune()
    elif run_mode == 'eval':
        task.evaluate()
    elif run_mode == 'predict':
        result = task.predict(input_data=predict_data)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='train', required=True, help='set run mode for model.')
    parser.add_argument('--task', default='text_generation', required=True, help='set task type.')
    parser.add_argument('--model_type', default='gpt2', required=True, help='set model type.')
    parser.add_argument('--train_dataset', default=None, help='set train dataset.')
    parser.add_argument('--eval_dataset', default=None, help='set eval dataset.')
    parser.add_argument('--predict_data', default='hello!', help='input data used to predict.')
    parser.add_argument('--pet_method', default='', help="set finetune method, now support type: ['', 'lora']")
    parser.add_argument('--data_parallel', default=1, type=int, help='set data parallel number. Default: None')
    parser.add_argument('--model_parallel', default=1, type=int, help='set model parallel number. Default: None')
    parser.add_argument('--pipeline_parallel', default=1, type=int, help='set pipeline parallel number. Default: None')
    parser.add_argument('--micro_size', default=1, type=int, help='set micro batch number. Default: None')
    parser.add_argument('--optimizer_parallel', default=False, type=bool,
                        help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()
    main(run_mode=args.run_mode,
         task=args.task,
         model_type=args.model_type,
         pet_method=args.pet_method,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         dp=args.data_parallel,
         mp=args.model_parallel,
         pp=args.pipeline_parallel,
         micro_size=args.micro_size,
         op=args.optimizer_parallel)
```

- 单机多卡标准启动脚本：`run_distribute_single_node.sh`

```bash
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

if [ $# != 4 ]
then
echo "Usage Help: bash run_distribute_single_node.sh [EXECUTE_ORDER] [RANK_TABLE_PATH]  [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices In Single Machine"
exit 1
fi

check_real_path(){
if [ "${1:0:1}" == "/" ]; then
  echo "$1"
else
  echo "$(realpath -m $PWD/$1)"
fi
}

EXECUTE_ORDER=$1
RANK_TABLE_PATH=$(check_real_path $2)
DEVICE_RANGE=$3

DEVICE_RANGE_LEN=${#DEVICE_RANGE}
DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
PREFIX=${DEVICE_RANGE%%","*}
INDEX=${#PREFIX}
START_DEVICE=${DEVICE_RANGE:0:INDEX}
END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}

if [ ! -f $RANK_TABLE_PATH ]
then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
exit 1
fi

if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
  echo "error: start_device=$START_DEVICE is not a number"
exit 1
fi

if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
  echo "error: end_device=$END_DEVICE is not a number"
exit 1
fi

ulimit -u unlimited

export RANK_SIZE=$4
export RANK_TABLE_FILE=$RANK_TABLE_PATH

shopt -s extglob

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export DEVICE_ID=${i}
  export RANK_ID=$((i-START_DEVICE))
  mkdir -p ./output/log/rank_$RANK_ID
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
done

shopt -u extglob
```

- 多机多卡标准启动脚本：`run_distribute_multi_node.sh`

```bash
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

if [ $# != 4 ]
then
echo "Usage Help: bash run_distribute_multi_node.sh [EXECUTE_ORDER] [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE]"
exit 1
fi

check_real_path(){
if [ "${1:0:1}" == "/" ]; then
  echo "$1"
else
  echo "$(realpath -m $PWD/$1)"
fi
}

EXECUTE_ORDER=$1
RANK_TABLE_PATH=$(check_real_path $2)
DEVICE_RANGE=$3

DEVICE_RANGE_LEN=${#DEVICE_RANGE}
DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
PREFIX=${DEVICE_RANGE%%","*}
INDEX=${#PREFIX}
START_DEVICE=${DEVICE_RANGE:0:INDEX}
END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}

if [ ! -f $RANK_TABLE_PATH ]
then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
exit 1
fi

if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
  echo "error: start_device=$START_DEVICE is not a number"
exit 1
fi

if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
  echo "error: end_device=$END_DEVICE is not a number"
exit 1
fi

ulimit -u unlimited

export RANK_SIZE=$4
export RANK_TABLE_FILE=$RANK_TABLE_PATH

shopt -s extglob
for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export RANK_ID=${i}
  export DEVICE_ID=$((i-START_DEVICE))
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  mkdir -p ./output/log/rank_$RANK_ID
  $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
done

shopt -u extglob
```

- 分布式并行执行`distribute_task.py`样例：需提前生成`RANK_TABLE_FILE`，同时`distribute_task.py`中默认使用**半自动并行模式**。

**注意单机时使用{single}，多机时使用{multi}，命令中`RANK_TABLE_FILE`文件名需改为实际文件名**

```shell
# 分布式训练
bash run_distribute_{single/multi}_node.sh "python distribute_task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode train --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_127.0.0.1.json [0,4] 4

# 分布式评估
bash run_distribute_{single/multi}_node.sh "python distribute_task.py --task text_generation --model_type gpt2 --eval_dataset ./eval --run_mode eval --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_127.0.0.1.json [0,4] 4

# 分布式微调
bash run_distribute_{single/multi}_node.sh "python distribute_task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode finetune --pet_method lora --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_127.0.0.1.json [0,4] 4

# 分布式推理（不支持流水并行,试用特性）
bash run_distribute_{single/multi}_node.sh "python distribute_task.py --task text_generation --model_type gpt2 --predict_data 'hello!' ./train --run_mode predict --pet_method lora --data_parallel 1 --model_parallel 2" hccl_4p_0123_127.0.0.1.json [0,4] 4
```