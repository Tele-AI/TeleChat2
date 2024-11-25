# InternLM

## 模型描述

InternLM ，即书生·浦语大模型，是由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。包含面向实用场景的70亿和200亿参数基础模型与对话模型 （InternLM-7B/20B）。模型具有以下特点：

- 使用上万亿高质量语料，建立模型超强知识体系；
- 支持8k语境窗口长度，实现更长输入与更强推理体验；
- 通用工具调用能力，支持用户灵活自助搭建流程；

本仓库目前能够支持上述特性1，暂未支持特性2、3。

本仓库支持InternLM-7B的微调和InternLM-Chat-7B/20B的推理。由于InternLM与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

``` text
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                      |      Task       | Datasets | SeqLength |  Phase   |   Performance   |
|:--------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:---------------:|
| [InternLM_7B](./finetune_internlm_7b.yaml)  | text_generation |  alpaca  |   2048    | Finetune | 3250 tokens/s/p |
| [InternLM_7B](./predict_internlm_7b.yaml)   | text_generation |    /     |   2048    | Predict  |   62 tokens/s   |
| [InternLM_20B](./predict_internlm_20b.yaml) | text_generation |    /     |   2048    | Predict  |  296 tokens/s   |

## 模型文件

`InternLM` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    research/internlm
        ├── internlm_tokenizer.py       # tokenizer
        ├── internlm_transformer.py     # transformer层实现
        ├── internlm_config.py          # 模型config
        └── internlm.py                 # 模型实现
    ```

2. 模型配置：

    ```text
    research/internlm
        ├── finetune_internlm_7b.yaml             # InternLM-7B全参微调Atlas 800T A2启动配置
        ├── finetune_internlm_7b_lora.yaml        # InternLM-7B lora低参微调Atlas 800T A2启动配置
        ├── predict_internlm_7b.yaml              # InternLM-7B推理Atlas 800T A2启动配置
        └── predict_internlm_20b.yaml             # InternLM-20B推理Atlas 800T A2启动配置
    ```

3. 预处理脚本和任务启动脚本：

    ```text
    research/internlm
        ├── alpaca_data_preprocess.py     # alpaca数据集预处理
        ├── wiki_data_preprocess.py       # wikitext2数据集预处理
        ├── convert_weight.py             # hf->mf权重转换
        ├── convert_reversed.py           # mf->hf权重转换
        └── run_internlm.py               # 高阶接口使用脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持7b，20b单机单卡推理，7b的全参微调支持单机八卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供`Wikitext2`作为**预训练**数据集，`alpaca`作为**微调**数据集。

| 数据集名称               |                适用模型                 |   适用阶段   |                                                         下载链接                                                          |
|:--------------------|:-----------------------------------:|:--------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2           | internlm-7b <br/> internlm20b <br/> | Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| alpaca              | internlm-7b <br/> internlm20b <br/> | Finetune |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |
| alpaca-gpt4-data-zh | internlm-7b <br/> internlm20b <br/> | Finetune |        [Link](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh/blob/main/alpaca_gpt4_data_zh.json)         |

下载数据集后，使用预处理脚本生成mindrecord训练数据：

- `WikiText2`数据集预处理指令示例：

```shell
cd research/internlm
python wiki_data_preprocess.py \
 --mindrecord_schema internlm_wiki \
 --input_glob {path}/wikitext-2/wiki.train.tokens \
 --output_file {path}/wiki_processed/wiki.mindrecord \
 --model_file {path}/tokenizer.model \
 --seq_length 2048 \
 --min_length 50  # 过滤token长度小于min_length的数据, default=50
```

- `Alpaca`数据集预处理指令示例：（同时适用于alpaca_data和alpaca-gpt4-data-zh数据集）

```shell
cd research/internlm
python alpaca_data_preprocess.py \
 --mindrecord_schema internlm_alpaca \
 --input_glob {path}/alpaca_data.json \
 --output_file {path}/alpaca_processed/alpaca.mindrecord \
 --model_file {path}/tokenizer.model \
 --seq_length 2048
```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。Base用于微调，Chat用于推理。

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/tokenizer.model)

| 模型名称              |                                                 MindSpore权重                                                  |                       HuggingFace权重                       |
|:------------------|:------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------:|
| InternLM-7B-Base  |   [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm.ckpt)    |    [Link](https://huggingface.co/internlm/internlm-7b)    |
| InternLM-7B-Chat  | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm-chat.ckpt) | [Link](https://huggingface.co/internlm/internlm-chat-7b)  |                                                                                                            |  |
| InternLM-20B-Base |                                                      /                                                       |   [Link](https://huggingface.co/internlm/internlm-20b)    |
| InternLM-20B-Chat |                                                      /                                                       | [Link](https://huggingface.co/internlm/internlm-chat-20b) |

#### 模型权重转换

原始权重下载完成后，运行如下转换脚本，将Hugging Face的权重转换为完整的ckpt权重。

1. 安装权重转换环境依赖

   ```shell
   pip install torch==2.0.0 transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. 执行权重转换脚本

   ```shell
   python research/internlm/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME

   # 参数说明
   model:       模型名称
   input_path:  下载HuggingFace权重的文件夹路径
   output_path: 转换后的MindSpore权重文件保存路径
   ```

## 微调

### 全参微调

MindFormers提供`InternLM-7B`全参微调示例，使用`alpaca_data`数据集，可通过[数据集下载](#数据集下载)获取数据集。

#### 单机训练

以`InternLM-7B`单机8卡全参微调为例，设置`seq_length=2048`，使用`research/internlm/finetune_internlm_7b.yaml`配置文件。

1. 模型权重使用说明

   若输入权重为完整权重，可直接设置`--load_checkpoint internlm_7b.ckpt`和`--auto_trans_ckpt True`进行权重自动切分；

   若加载分布式权重，则需要输入分布式权重文件夹路径，并按照`model_dir/rank_0/xxx.ckpt`格式存放，设置`--load_checkpoint model_dir`。

   权重切分与合并详细说明可见[参考文档](../../docs/feature_cards/Transform_Ckpt.md)。

2. 启动微调任务，以单机八卡为例，指令如下：

   ```shell
   bash scripts/msrun_launcher.sh "research/internlm/run_internlm.py \
    --config research/internlm/finetune_internlm_7b.yaml \
    --load_checkpoint path/to/ckpt \
    --train_dataset {path}/train_data \
    --run_mode finetune \
    --use_parallel True \
    --auto_trans_ckpt True" 8
   ```

#### 多机训练

多机多卡训练可以参考[多机多卡启动方式](../../README.md#多机多卡)。

### LoRA微调

MindFormers提供`InternLM-7B`LoRA微调示例，使用`alpaca-gpt4-data-zh`数据集，可通过[数据集下载](#数据集下载)获取数据集。

LoRA微调使用配置文件`research/internlm/finetune_internlm_7b_lora.yaml`

- 单卡启动指令如下：

  ```shell
  python research/internlm/run_internlm.py \
   --config research/internlm/finetune_internlm_7b_lora.yaml \
   --load_checkpoint path/to/ckpt \
   --train_dataset {path}/train_data \
   --run_mode finetune \
   --use_parallel False \
   --auto_trans_ckpt False \
   --device_id 0
  ```

- 多卡启动以单机8卡为例，指令如下：

```shell
bash scripts/msrun_launcher.sh "research/internlm/run_internlm.py \
 --config research/internlm/finetune_internlm_7b_lora.yaml \
 --load_checkpoint path/to/ckpt \
 --train_dataset {path}/train_data \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

## 推理

MindFormers提供`InternLM`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/internlm/run_internlm_predict.sh PARALLEL CONFIG_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

```shell
# internlm 7b
bash scripts/examples/internlm/run_internlm_predict.sh single \
 research/internlm/predict_internlm_7b.yaml \
 path/to/InternLM_7B_Chat.ckpt \
 path/to/tokenizer.model

# 推理结果
# <s> <|User|>:你是谁？<eoh>
# <|Bot|>我是一名人工智能助手，我的名称是书生·浦语。我由上海人工智能实验室开发...
# <s><s><|User|>:帮助我制定一份去上海的旅游攻略<eoh>
# <|Bot|>好的，以下是一份上海旅游攻略：
# 第一天：
# - 上午：到达上海，前往酒店休息...

# internlm 20b
bash scripts/examples/internlm/internlm_predict.sh single \
 research/internlm/predict_internlm_20b.yaml \
 path/to/InternLM_20B_Chat.ckpt \
 path/to/tokenizer.model

# 推理结果
# <s> <|User|>:你是谁？<eoh>
# <|Bot|>我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我能够回答问题...
# <s><s><|User|>:帮助我制定一份去上海的旅游攻略<eoh>
# <|Bot|>好的，以下是一份简单的上海旅游攻略：
# 第一天：
# 上午：
# - 上午到达上海，先去酒店办理入住手续，放下行李...
```

### 多卡推理

以`InternLM`2卡推理为例，推理结果与单卡推理相同。

```shell
# internlm 7b
bash scripts/examples/internlm/run_internlm_predict.sh parallel \
 research/internlm/predict_internlm_7b.yaml \
 path/to/InternLM_7B_Chat.ckpt \
 path/to/tokenizer.model 2

# internlm 20b
bash scripts/examples/internlm/run_internlm_predict.sh parallel \
 research/internlm/predict_internlm_20b.yaml \
 path/to/InternLM_20B_Chat.ckpt \
 path/to/tokenizer.model 2
```