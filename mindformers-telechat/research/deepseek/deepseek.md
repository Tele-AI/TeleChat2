# DeepSeek Coder

## 模型描述

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                       |      Task       |  Datasets   | SeqLength |  Phase   |  Performance   |
|:---------------------------------------------|:---------------:|:-----------:|:---------:|:--------:|:--------------:|
| [deepseek-33b](./predict_deepseek_33b.yaml)  | text_generation |      -      |   16384   | Predict  |  292 tokens/s  |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                       |      Task       |  Datasets   | SeqLength |  Phase   |  Performance   |
|:---------------------------------------------|:---------------:|:-----------:|:---------:|:--------:|:--------------:|
| [deepseek-33b](./finetune_deepseek_33b.yaml) | text_generation | code_alpaca |   4096    | Finetune | 572 tokens/s/p |

## 模型文件

`deepseek_33b` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型配置：

    ```text
    research/deepseek
        ├── finetune_deepseek_33b.yaml     # 全参微调启动配置
        ├── pretrain_deepseek_33b_16k.yaml # 预训练启动配置
        ├── predict_deepseek_33b.yaml      # huggingface转ckpt
        └── deepseek_preprocess.py         # 在线推理启动配置
    ```

2. 数据预处理脚本：

   ```text
    research/deepseek
        ├── alpaca_converter.py           # code_alpaca数据集格式转换脚本
        └── deepseek_preprocess.py        # 数据集预处理脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持33b单机4卡推理，全参微调至少需要2机16卡，预训练至少需要2机16卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供`Wikitext-103`作为[预训练](#预训练)数据集，`code_alpaca`作为[全参微调](#全参微调)数据集。

| 数据集名称        |     适用模型     |   适用阶段   |                                            下载链接                                            |
|:-------------|:------------:|:--------:|:------------------------------------------------------------------------------------------:|
| Wikitext-103 | deepseek_33b | Pretrain | [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) |
| code_alpaca  | deepseek_33b | Finetune |  [Link](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json)   |

数据预处理中所用的`tokenizer.json`可以通过[链接](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json)进行下载。

- **Wikitext-103 数据预处理**

  使用`research/deepseek/deepseek_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```bash
  python deepseek_preprocess.py \
   --dataset_type 'wiki' \
   --input_glob /path/wiki.train.tokens \
   --model_file /path/tokenizer.json \
   --seq_length 16384 \
   --output_file /path/wiki.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.train.tokens的文件路径
  model_file:   vocab.json文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

- **code_alpaca 数据预处理**

  执行`research/deepseek/alpaca_converter.py`，将原始数据集转换为指定格式。

  ```shell
  python alpaca_converter.py \
   --data_path path/alpaca_data.json \
   --output_path /path/alpaca-data-messages.json

  # 参数说明
  data_path:   输入下载后code_alpaca的文件路径
  output_path: 输出转换后文件的保存路径
  ```

  执行`research/deepseek/deepseek_preprocess.py`，进行数据预处理和Mindrecord数据生成。

  ```shell
  python deepseek_preprocess.py \
   --dataset_type qa \
   --input_glob /path/alpaca-data-messages.json \
   --model_file /path/tokenizer.json \
   --seq_length 4096 \
   --output_file /path/alpaca-messages.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   转换后的alpaca的文件路径
  model_file:   tokenizer.json文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

### 模型权重下载

MindFormers提供下载HuggingFace官方权重的下载链接，用户可通过链接下载权重并经过[模型权重转换](#模型权重转换)后进行使用，`tokenizer.json`文件也在链接中下载。

词表下载链接：[tokenizer.json](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json)

| 模型名称         | MindSpore权重 |                             HuggingFace权重                              |
|:-------------|:-----------:|:----------------------------------------------------------------------:|
| deepseek-33b |      -      | [Link](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct) |

#### 模型权重转换

执行`research/deepseek/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python research/deepseek/convert_weight.py \
 --torch_ckpt_path TORCH_CKPT_PATH \
 --mindspore_ckpt_path MS_CKPT_NAME

# 参数说明
torch_ckpt_path: 下载HuggingFace权重文件夹路径
mindspore_ckpt_path: 转换后的MindSpore权重文件保存路径
```

- **模型权重切分与合并**

  从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

  通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

  以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 预训练

MindFormers提供`deepseek-33b`多机预训练示例，，使用`Wikitext-103`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

`deepseek-33b`多机预训练使用配置文件`pretrain_deepseek_33b_16k.yaml`，不支持单机进行预训练任务。

多机多卡训练需要不同节点上执行启动命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README.md#三使用指南)。

执行如下命令启动2机16卡预训练任务：

```shell
# 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config research/deepseek/pretrain_deepseek_33b_16k.yaml \
 --train_dataset_dir /path/wiki.mindrecord \
 --use_parallel True \
 --run_mode train" \
 16 8 {ip_addr} 8118 0 output/msrun_log False 300

# 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config research/deepseek/pretrain_deepseek_33b_16k.yaml \
 --train_dataset_dir /path/wiki.mindrecord \
 --use_parallel True \
 --run_mode train" \
 16 8 {ip_addr} 8118 1 output/msrun_log False 300

# 参数说明
config:            配置文件路径
run_mode:          运行模式, 预训练时设置为train
train_dataset_dir: 训练数据集路径
use_parallel:      是否开启并行训练
```

> 注：此模型暂不支持配置`context_parallel`，因此暂不支持长序列。

## 全参微调

MindFormers提供`deepseek-33b`多机多卡微调示例，使用`code_alpaca`数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

`deepseek-33b`多机微调使用配置文件`finetune_deepseek_33b.yaml`，不支持单机进行微调任务。

1. 生成多机分布式权重

   如果使用共享存储，可以将模型完整权重放在共享存储内，同时设置配置文件或脚本参数`auto_trans_ckpt=True`，使用权重自动转换功能。

   如果不使用共享存储，可以参考[多机多卡权重转换](../../docs/feature_cards/Transform_Ckpt.md#物理机多机多卡训练)完成分布式权重转换后拉起预训练任务。

2. 执行命令启动2机16卡微调任务，以使用共享存储为例

   修改模型配置文件`research/deepseek/finetune_deepseek_33b.yaml`中分布式并行策略。

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 2
     micro_batch_num: 128
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

   多机多卡训练需要不同节点上执行启动命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README.md#三使用指南)。

   ```shell
   # 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config research/deepseek/finetune_deepseek_33b.yaml \
    --load_checkpoint /path/deepseek_33b.ckpt \
    --train_dataset_dir /path/alpaca-messages.mindrecord \
    --use_parallel True \
    --auto_trans_ckpt True \
    --run_mode finetune" \
    16 8 {ip_addr} 8118 0 output/msrun_log False 300

   # 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config research/deepseek/finetune_deepseek_33b.yaml \
    --load_checkpoint /path/deepseek_33b.ckpt \
    --train_dataset_dir /path/alpaca-messages.mindrecord \
    --use_parallel True \
    --auto_trans_ckpt True \
    --run_mode finetune" \
    16 8 {ip_addr} 8118 1 output/msrun_log False 300

   # 参数说明
   config:            配置文件路径
   load_checkpoint:   模型权重文件路径
   train_dataset_dir: 训练数据集路径
   use_parallel:      是否开启并行训练
   auto_trans_ckpt:   自动权重转换开关
   run_mode:          运行模式, 微调时设置为finetune
   ```

## 推理

MindFormers提供`deepseek-33b`推理示例，使用配置文件`predict_deepseek_33b.yaml`，仅支持多卡推理。

1. 修改配置文件`research/deepseek/predict_deepseek_33b.yaml`

   ```yaml
   processor:
     tokenizer:
       vocab_file: None
       tokenizer_file: "/path/tokenizer.json"
   ```

2. 执行4卡推理命令：

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config research/deepseek/predict_deepseek_33b.yaml \
    --run_mode predict \
    --predict_data '#write a quick sort algorithm' \
    --predict_length 100 \
    --use_parallel True \
    --use_past True" 4

   # 推理结果
   # #write a quick sort algorithm
   # def quick_sort(arr):
   #     if len(arr) <= 1:
   #         return arr
   #     pivot = arr[len(arr) // 2]
   #     left = [x for x in arr if x < pivot]
   #     middle = [x for x in arr if x == pivot]
   #     right = [x for x in arr if x > pivot]
   #    return quick_sort(left) + middle + quick_sort(right)
   #
   # print(quick_sort([3,6,8,10,1,2,1]))
   # # Prints "[1, 1, 2, 3, 6, 8, 10]"
   ```
