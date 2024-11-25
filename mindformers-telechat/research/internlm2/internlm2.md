# InternLM2

## 模型描述

第二代浦语模型，InternLM2 的基础模型具备以下的技术特点：

有效支持20万字超长上下文：模型在20万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。
综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码等方面的能力提升显著。

本仓库支持`InternLM2-7B`与`InternLM2-chat-20b`的推理。由于InternLM2与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

## 模型文件

`InternLM2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现

    ```text
    research/internlm2
      ├── internlm2_tokenizer.py       # tokenizer
      ├── internlm2_transformer.py     # transformer层实现
      ├── internlm2_config.py          # 模型config
      ├── internlm2.py                 # 模型实现
      └── internlm2_interleave.py
    ```

2. 模型配置

    ```text
    research/internlm2
      ├── predict_internlm2_20b.yaml  # InternLM2-chat-20B推理Atlas 800T A2启动配置
      ├── predict_internlm2_chat_7b.yaml   # InternLM2-7B推理Atlas 800T A2启动配置
      └── finetune_internlm2_7b.yaml       # InternLM2-7B微调Atlas 800T A2启动配置
    ```

3. 预处理脚本和任务启动脚本

    ```text
    research/internlm2
      ├── convert_weight.py             # hf->mf权重转换
      ├── convert_reversed.py           # mf->hf权重转换
      └── run_internlm2.py              # 高阶接口使用脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

## 数据及权重准备

### 数据集下载

MindFormers提供**alpaca**作为[微调](#微调)数据集。

| 数据集名称     |                          适用模型                          |          适用阶段           |                                                         下载链接                                                          |
|:----------|:------------------------------------------------------:|:-----------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| alpaca    |                      InternLM2-7b                      |        Finetune         |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |

下载数据集后，使用预处理脚本`research/internlm2/alpaca_data_preprocess.py`生成mindrecord训练数据:

```shell
python alpaca_data_preprocess.py \
  --mindrecord_schema internlm2_alpaca \
  --input_glob {path}/alpaca_data.json \
  --output_file {path}/alpaca_processed/alpaca.mindrecord \
  --model_file {path}/tokenizer.model \
  --seq_length 2048
```

### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。Base用于微调，Chat用于推理。

词表下载链接：[tokenizer.model](https://huggingface.co/internlm/internlm2-7b/blob/main/tokenizer.model)

|       模型名称        | MindSpore权重 |                        HuggingFace权重                       |
|:-----------------:|:-----------:|:--------------------------------------------------------------------:|
|   InternLM2-7b    |      \      |         [link](https://huggingface.co/internlm/internlm2-7b)         |
| InternLM2-chat-7B |      \      |       [link](https://huggingface.co/internlm/internlm2-chat-7b)      |
|   InternLM2-chat-20b   |      \      | [link](https://huggingface.co/internlm/internlm2-chat-20b)  |

### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py \
  --model        internlm2 \
  --input_path   TORCH_CKPT_DIR \
  --output_path  {path}/MS_CKPT_NAME \
  --qkv_concat   True

  # 参数说明
  input_path:   huggingface权重保存目录路径
  output_path:  权重保存文件名, 可以指定自定义保存路径
  qkv_concat:   是否qkv融合
```

## 微调

MindFormers提供`InternLM2-7b`的微调示例， 过程中使用alpaca数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

执行msrun启动脚本，进行8卡分布式微调

```shell
bash scripts/msrun_launcher.sh "research/internlm2/run_internlm2.py \
  --config research/internlm2/finetune_internlm2_7b.yaml \
  --train_dataset path/to/tain_dataset \
  --load_checkpoint path/to/checkpoint \
  --run_mode finetune \
  --use_parallel True" 8

  # 参数说明
  config:           模型配置文件路径
  train_dataset:    微调数据集路径
  load_checkpoint:  模型权重文件路径
  run_mode:         运行模式
  use_parallel:     是否开启并行
```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage

#### 多机训练

多机多卡训练可以参考[多机多卡启动方式](https://gitee.com/mindspore/mindformers/blob/dev/README.md#%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1)。

## 推理

MindFormers提供`InternLM2`的快速推理脚本，脚本主要通过`generate`高阶接口实现，支持单卡、多卡以及多batch推理。`InternLM2-7B`的推理流程与`InternLM2-chat-20b`相同。仅需替换配置文件。

```shell
# 脚本使用
bash scripts/examples/internlm2/run_internlm2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM TOKENIZER_PATH

# 参数说明
PARALLEL:        是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH:     模型配置文件路径
CKPT_PATH:       模型权重文件路径
DEVICE_NUM:      使用卡数, 仅开启多卡推理时生效
TOKENIZER_PATH:  Tokenizer模型路径
```

### 单卡推理

```shell
bash scripts/examples/internlm2/run_internlm2_predict.sh \
  single \
  configs/internlm2/predict_internlm2_20b.yaml \
  path/to/internlm2_chat_20b.ckpt \
  1 \
  path/to/tokenizer.model
# input: Hi, pls intro yourself
# output: Hello! I'm an AI language model designed to assist and provide helpful responses to your inquiries. I don't have a personal identity or consciousness, but I'm here to help you with any questions or tasks you might have. How can I assist you today?
#
# input: Shanghai is a
# output：Shanghai is a city, not a person. It is the largest city in China and one of the most populous metropolitan areas in the world. It is located on the east coast of China......
#
# input: Huawei is a
# output: Huawei is a multinational technology company headquartered in Shenzhen, China. Founded in 1987, the company has grown to become one of the largest and most influential technology companies in the world, with a diverse range of products and services that span across various industries......
```

### 多卡推理

`InternLM2`多卡推理暂不支持`is_dynamic=True`。本示例以`InternLM2-chat-20b`8卡推理为例。

```shell
bash scripts/examples/internlm2/run_internlm2_predict.sh \
  parallel \
  configs/internlm2/predict_internlm2_20b.yaml \
  path/to/internlm2_chat_20b.ckpt \
  8 \
  path/to/tokenizer.model
# input: Hi, pls intro yourself
# output: Hello! I'm an AI language model designed to assist and provide helpful responses to your inquiries. I don't have a personal identity or consciousness, but I'm here to help you with any questions or tasks you might have. How can I assist you today?
#
# input: Shanghai is a
# output：Shanghai is a city, not a person. It is the largest city in China and one of the most populous metropolitan areas in the world. It is located on the east coast of China......
#
# input: Huawei is a
# output: Huawei is a multinational technology company headquartered in Shenzhen, China. Founded in 1987, the company has grown to become one of the largest and most influential technology companies in the world, with a diverse range of products and services that span across various industries......
```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage
