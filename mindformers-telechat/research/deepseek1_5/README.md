# DeepSeek Coder

## 模型描述

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek
Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                |      Task       | Datasets | SeqLength |  Phase  | Performance |
|:------------------------------------------------------|:---------------:|:--------:|:---------:|:-------:|:-----------:|
| [deepseek1.5-7b](deepseek1_5_7b/predict_deepseek_coder1_5_7b.yaml) | text_generation |    -     |   2048    | Predict | 60 tokens/s |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                 |      Task       |  Datasets   | SeqLength |  Phase   |  Performance   |
|:-------------------------------------------------------|:---------------:|:-----------:|:---------:|:--------:|:--------------:|
| [deepseek1.5-7b](deepseek1_5_7b/finetune_deepseek_coder1_5_7b.yaml) | text_generation | code_alpaca |   8192    | Finetune | 340 tokens/s/p |

## 模型文件

`deepseek-coder-7b-v1.5` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型配置：

    ```text
    deepseek1_5/deepseek1_5_7b
        ├── finetune_deepseek_coder1_5_7b.yaml     # 全参微调启动配置
        └── predict_deepseek_coder1_5_7b.yaml     # 在线推理启动配置
    ```

2. 模型相关脚本：

    ```text
    deepseek1_5
        ├── alpaca_converter.py                   # alpaca数据集格式转换脚本
        ├── convert_weight.py                     # hf->ms权重转换脚本
        ├── convert_reversed.py                   # ms->hf权重转换脚本
        └── deepseek_preprocess_1_5.py            # alpaca数据集格式转换脚本
    ```

## 环境及数据准备

### 安装环境

### 环境参数设置

```shell
export MS_DEV_RUNTIME_CONF="inline:true"
```

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#二mindformers安装)和[版本匹配关系](../../README_CN.md#三版本匹配关系)。

### 数据及权重准备

#### 数据集下载

当前提供code_alpaca_20k.json数据集的预处理和微调样例，用于对Deepseek-Coder-7B-v1.5-Instruct，Deepseek-Coder-7B-v1.5-Base模型进行微调。

| 数据集名称                           | 适用模型           |                                          适用阶段                                           | 下载链接                                                                                    |
|:--------------------------------|----------------|:---------------------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------|
| code_alpaca_20k.json | Deepseek-Coder-7B-v1.5-Instruct<br/>Deepseek-Coder-7B-v1.5-Base | finetune / lora | [Link](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) |

下载数据集后，需要先执行alpaca_converter.py脚本将数据集转换为alpaca-data-converted.json，再用deepseek_preprocess_1_5.py脚本进行数据预处理，将原始数据转换为mindrecord格式。数据预处理中所用的`tokenizer.json`可以通过[链接](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json)进行下载。

```python
python alpaca_converter.py \
  --data_path /path/code_alpaca_20k.json \
  --output_path /path/alpaca-data-converted.json

python deepseek_preprocess_1_5.py \
  --dataset_type qa \
  --input_glob /path/alpaca-data-converted.json  \
  --model_file /path/tokenizer.json \
  --seq_length 4096 \
  --output_file /path/alpaca_finetune_4k.mindrecord

参数说明：

- dataset_type: 固定值 qa
- Input_glob：待处理的数据集路径，具体到文件
- model_file：tokenizer文件，一般从huggingfce下载，具体到文件
- seq_length：词表长度，当前固定为4096，可以修改为16384等
- pad_token_id：空白填充ID，当前固定为100015
- output_file：输出文件，具体到文件，后缀为mindrecord
```

#### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，tokenizer.json文件也在链接中下载。

| 模型名称                            | MindSpore权重 |                             权重                             | 用途 |
|:--------------------------------|------------| :----------------------------------------------------------: |----|
| Deepseek-Coder-7B-v1.5-Instruct | - | [Link](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5/tree/main) | 推理 |
| Deepseek-Coder-7B-v1.5-Base     | - |       [Link](deepseek-ai/deepseek-coder-7b-base-v1.5)        | 微调 |

#### 模型权重转换

##### torch权重转mindspore权重

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```python
cd research
python convert_weight.py \
--model deepseek1_5 \
--input_path /path/ckpt \
--output_path MS_CKPT_NAME
```

- 参数说明
  input_path: 预训练权重文件所在的目录, 此参数必须, 且需要包含config.json相关模型文件
  output_path: 转换后的输出文件存放路径（`.ckpt`文件）, 此参数必须

##### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)*

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 自动转换权重

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../../docs/feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../../docs/feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 全参微调

执行前修改finetune_deepseek_coder1_5_7b.yaml文件的参数`tokenizer_file`为文件`tokenizer.json`路径

在项目根目录执行如下命令：

```shell
bash ./scripts/msrun_launcher.sh "./run_mindformer.py \
--config ./research/deepseek1_5/finetune_deepseek_coder1_5_7b.yaml \
--use_parallel True \
--load_checkpoint  ./ckpt_trans \
--run_mode train \
--train_data  ./dataset/train_data" \
8 8 127.0.0.1 9543 0 output/msrun_log False 3000;
```

参数说明：

- config: 固定路径，配置文件所在路径
- usr_parallel：固定值，True
- load_checkpoint：加载切分后权重的路径，具体到文件夹
- run_mode：固定值，train
- train_data：数据集所在位置，具体到文件夹

> 注：此模型暂不支持配置`context_parallel`，因此暂不支持长序列。

## 推理

使用推理功能时，目前仅支持多卡推理，推荐使用Deepseek-Coder-7B-v1.5-Instruct和Deepseek-Coder-7B-v1.5-Base权重，默认设置seq_length=4096，
模型权重以及tokenizer文件可参考模型权重下载。

### 参数配置

> 核查配置文件 `predict_deepseek_coder1_5_7b.yaml`。
> 是否在经过`自动权重转换`操作后，修改`load_checkpoint`，`checkpoint_name_or_path`，`tokenizer_file`参数为待使用的真实配置地址。
> 核查无误进行后续操作。

### 多卡推理

执行前修改predict_deepseek_coder1_5_7b.yaml文件的参数`auto_trans_ckpt`为`True`，参数`tokenizer_file`为文件`tokenizer.json`路径，
并配置参数`load_checkpoint`的路径为加载权重的路径

  ```shell
  bash scripts/msrun_launcher.sh "run_mindformer.py
  --config research/deepseek1_5/predict_deepseek_coder1_5_7b.yaml
  --run_mode=predict
  --predict_data 'write a quick sort algorithm.'
  --predict_length 100
  --use_parallel True
  --use_past True" 2
  # 运行结果：[{'text_generation_text': ["write a quick sort algorithm.\nI'm trying to write a quicksort algorithm in python. I'm having trouble with the partition part.\nHere is my code:\n def quicksort(arr):\n if len(arr) <= 1:\n return arr\n pivot = arr[len(arr) // 2]\n left = [x for x in arr if x < pivot]\n middle = [x for x in arr if x == pivot]\n right = [x for x in arr if x > pivot]\n return quicksort(left) + middle + quicksort(right)\n print(quicksort([3,6,8,10,1,2,1]))\n\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nYou can use the following code to fix the problem:\n def quicksort(arr):\n if len(arr) <= 1:\n return arr\n pivot = arr[len(arr) // 2]\n left = [x for x in arr if x < pivot]\n middle = [x for x in arr if x == pivot]\n right = [x for x in arr if x > pivot]\n return quicksort(left) + middle + quicksort(right)\n print(quicksort([3,6,8,10,1,2,1]))\n\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe problem is that the pivot is not being used to split the array into two halves. I'm not sure what I'm doing wrong.\nThe"]}]
  ```