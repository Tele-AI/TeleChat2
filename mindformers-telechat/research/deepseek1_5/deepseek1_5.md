# DeepSeek Coder

## 模型描述

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek
Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                |      Task       | Datasets | SeqLength |  Phase  | Performance |
|:------------------------------------------------------|:---------------:|:--------:|:---------:|:-------:|:-----------:|
| [deepseek1.5-7b](./predict_deepseek_coder1_5_7b.yaml) | text_generation |    -     |   2048    | Predict | 60 tokens/s |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                 |      Task       |  Datasets   | SeqLength |  Phase   |  Performance   |
|:-------------------------------------------------------|:---------------:|:-----------:|:---------:|:--------:|:--------------:|
| [deepseek1.5-7b](./finetune_deepseek_coder1_5_7b.yaml) | text_generation | code_alpaca |   8192    | Finetune | 340 tokens/s/p |

## 模型文件

`deepseek-coder-7b-v1.5` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型配置：

    ```text
    deepseek1_5
        ├── deepseek_preprocess_1_5.py            # 此表转换脚本
        ├── finetune_deepseek_coder1_5_7b.yaml     # 全参微调启动配置
        └── predict_deepseek_coder1_5_7b.yaml     # 在线推理启动配置
    ```

2. 数据预处理脚本：

在`deepseek1_5`目录执行如下命令进行数据预处理

```python
python deepseek_preprocess_1_5.py \
--dataset_type qa \
--input_glob '' \
--model_file '' \
--seq_length 4096 \
--pad_token_id 100015 \
--output_file ./output/alpaca_finetune_4k.mindrecord
```

参数说明：

- dataset_type: 固定值 qa
- Input_glob：待处理的数据集路径，具体到文件
- model_file：tokenizer文件，一般从huggingfce下载，具体到文件
- seq_length：词表长度，当前固定为4096，可以修改为16384等
- pad_token_id：空白填充ID，当前固定为100015
- output_file：输出文件，具体到文件，后缀为mindrecord

## 环境及数据准备

### 安装环境

### 环境参数设置

```shell
export MS_DEV_RUNTIME_CONF="inline:false"
```

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 数据及权重准备

#### 数据集下载

[https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json]()

### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，tokenizer.json文件也在链接中下载。

| 模型名称                        |                             权重                             | 用图 |
| :------------------------------ | :----------------------------------------------------------: | ---- |
| deepseek-coder-7b-v1.5-instruct | [Link](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5/tree/main) | 推理 |
| Deepseek-coder-7b-v1.5-base     |       [Link](deepseek-ai/deepseek-coder-7b-base-v1.5)        | 微调 |

#### 模型权重转换

##### torch权重转mindspore权重

执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```python
cd research
python mindformers/convert_weight.py --model deepseek --input_path TORCH_CKPT_PATH --output_path MS_CKPT_NAME
```

- 参数说明
  input_path: 预训练权重文件所在的目录, 此参数必须
  output_path: 转换后的输出文件存放路径（`.ckpt`文件）, 此参数必须

##### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)*

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 自动转换权重

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../../docs/feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../../docs/feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 全参微调

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

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合算大算子降低推理时延，有效¡提升网络吞吐量。

### 基于高阶接口推理

#### 参数配置

> 核查配置文件 `predict_deepseek_coder1_5_7b.yaml`。
> 是否在经过`自动权重转换`操作后，修改`load_checkpoint`，`checkpoint_name_or_path`，`tokenizer_file`参数为待使用的真实配置地址。
> 核查无误进行后续操作。

#### 单卡推理

  ```shell
  cd {mindformers根目录}
  bash scripts/msrun_launcher.sh "run_mindformer.py --config research/deepseek1_5/predict_deepseek_coder1_5_7b.yaml --run_mode=predict --predict_data '#write a quick sort algorithm' --predict_length 100 --use_parallel False --use_past True" 1
  # 运行结果：[{'text_generation_text': ['#write a quick sort algorithm\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a merge sort algorithm\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result += left[i:]\n    result += right[j:]\n    return result\n\nprint(merge_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a bubble sort algorithm\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, '']}]
  ```

#### 多卡推理

  ```shell
  cd {mindformers根目录}
  bash scripts/msrun_launcher.sh "run_mindformer.py --config research/deepseek1_5/predict_deepseek_coder1_5_7b.yaml --run_mode=predict --predict_data '#write a quick sort algorithm' --predict_length 100 --use_parallel True --use_past True" 2
  # 运行结果：[{'text_generation_text': ['#write a quick sort algorithm\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a merge sort algorithm\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result += left[i:]\n    result += right[j:]\n    return result\n\nprint(merge_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a bubble sort algorithm\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, '']}]
  ```