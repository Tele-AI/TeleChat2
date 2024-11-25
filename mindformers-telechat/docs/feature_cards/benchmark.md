# benchmark工具

## 概述

大模型训练benchmark，旨在提供高效的工具，支撑用户实现快捷部署。
用户提供模型代码、权重、分词模型、配置文件、训练数据等依赖的信息，即可快捷启动模型训练、微调任务。当前部分训练数据已经支持在线模式，随着开源社区的建设，模型训练、微调所需的各项信息将逐步支持在线模式。当前适配Llama2、Llama3.1、mixtral等模型，后续将持续适配更多模型。

## 使用介绍

用户只需提供必要的数据、配置文件、权重等信息，执行[pretrain_tool.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/benchmark/pretrain_tool.sh)脚本即可启动相应训练、微调任务。其中数据支持原始数据、预处理之后的数据，部分数据支持在线模式，即只提供数据集名称即可。脚本参数介绍如下：

| 参数              | 简写 | 单机是否必选 | 多机是否必选 | 默认值           | 说明                                                         |
| ----------------- | ---- | ------------ | ------------ | ---------------- | ------------------------------------------------------------ |
| model_name_or_dir | n    | √            | √            |                  | 模型路径，包含模型配置及权重(非必须)                         |
| pretrain_data     | i    | √            | √            |                  | 训练数据支持数据类型：<br>-原始数据集<br>-预处理后的数据集<br>-数据集名称（当前仅支持wikitext2） |
| worker_num        | w    | ×            | √            | 8                | 所有节点中使用计算卡的总数                                   |
| local_worker      | l    | ×            | √            | 8                | 当前节点中使用计算卡的数量                                   |
| master_addr       | a    | ×            | √            | 127.0.0.1        | 指定分布式启动主节点的ip                                     |
| master_port       | p    | ×            | √            | 8118             | 指定分布式启动绑定的端口号                                   |
| node_rank         | r    | ×            | √            | 0                | 指定当前节点的rank id                                        |
| log_dir           | g    | ×            | √            | output/msrun_log | 日志输出路径，若不存在则递归创建                             |
| join              | j    | ×            | √            | FALSE            | 是否等待所有分布式进程退出                                   |
| cluster_time_out  | t    | ×            | √            | 7200             | 分布式启动的等待时间，单位为秒                               |

## 使用示例

### 使用在线数据执行训练

以Llama2-7B为例，将模型训练所需的配置文件及对应的`tokenizer.model`放入`${model_pah}`目录，执行如下命令：

```shell
cd scripts/benchmark

bash pretrain_tool.sh -n ${model_pah} -i wikitext2
```

### 使用原始离线数据执行训练

以Llama2-7B为例，将模型训练所需的配置文件及对应的`tokenizer.model`放入`${model_pah}`目录，将训练数据`wiki.train.tokens`放入`${data_pah}`目录，执行如下命令：

```shell
cd scripts/benchmark

bash pretrain_tool.sh -n ${model_pah} -i ${data_pah}/wiki.train.tokens
```

### 使用预处理离线数据执行微调

以Llama3.1-8B为例，将模型训练所需的配置文件和权重`ckpt`文件放入`${model_pah}`，将训练数据`alpaca-fastchat8192.mindrecord`放入`${data_pah}`，执行如下命令：

```shell
cd scripts/benchmark

bash pretrain_tool.sh -n ${model_pah} -i ${data_pah}/alpaca-fastchat8192.mindrecord
```

### 使用预处理离线数据执行多机训练

以Mixtral为例，在两台机器上，将模型训练所需的配置文件和权重`ckpt`文件放入`${model_pah}`目录，将训练数据`alpaca-fastchat8192.mindrecord`放入`${data_pah}`目录，然后分别执行下述命令。

* 机器0

  ```shell
  cd scripts/benchmark

  bash pretrain_tool.sh -n ${model_pah} -i ${data_pah}/wiki.train.tokens -w 16 -l 8 -a ${master_addr} -p 8118 -r 0 -o output/msrun_log -j False -t 300
  ```

* 机器1

  ```shell
  cd scripts/benchmark

  bash pretrain_tool.sh -n ${model_pah} -i ${data_pah}/wiki.train.tokens -w 16 -l 8 -a ${master_addr} -p 8118 -r 1 -o output/msrun_log -j False -t 300
  ```

两台机器的命令主要是node_rank（即 -r）参数不同。
