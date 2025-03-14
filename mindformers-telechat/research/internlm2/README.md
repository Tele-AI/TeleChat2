# InternLM2

## 模型描述

第二代浦语模型，InternLM2 的基础模型具备以下的技术特点：

有效支持20万字超长上下文：模型在20万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。
综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码等方面的能力提升显著。

本仓库支持`InternLM2-7B`与`InternLM2-20B`的微调、推理。由于InternLM2与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

本仓库中InternLM2-20B的微调部分由“天翼云智算团队”贡献。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                           |      Task       | Datasets | SeqLength |  Phase   |   Performance   |
|:-------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:---------------:|
| [internlm2-20b](./finetune_internlm2_20b.yaml) | text_generation |    alpaca     |   4096    | Finetune  |  867.923 tokens/s/p  |

## 模型文件

`InternLM2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现

    ```text
      research/internlm2
      |-- internlm2_config.py     # 模型config
      |-- internlm2_model.py      # 模型实现
      `-- internlm2_tokenizer.py  # tokenizer
    ```

2. 模型配置

    ```text
      research/internlm2
      |-- internlm2_20b
      |   |-- finetune_internlm2_20b.yaml # InternLM2-20B微调启动配置
      |   `-- predict_internlm2_20b.yaml  # InternLM2-chat-20B推理启动配置
      `-- internlm2_7b
          |-- finetune_internlm2_7b.yaml  # InternLM2-7B微调启动配置
          `-- predict_internlm2_7b.yaml   # InternLM2-chat-7B推理启动配置
    ```

3. 预处理脚本和任务启动脚本

    ```text
      research/internlm2
      |-- alpaca_data_preprocess.py     # alpaca数据集预处理
      |-- convert_reversed.py           # mf->hf权重转换
      `-- convert_weight.py             # hf->mf权重转换
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

## 数据及权重准备

### 数据集下载

MindFormers提供**alpaca**作为[微调](#微调)数据集。

| 数据集名称     |                          适用模型                          |          适用阶段           |                                                         下载链接                                                          |
|:----------|:------------------------------------------------------:|:-----------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| alpaca    |                      InternLM2-7B <br/> InternLM2-20B                     |        Finetune         |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |

下载数据集后，使用预处理脚本`research/internlm2/alpaca_data_preprocess.py`生成mindrecord训练数据:

```shell
python alpaca_data_preprocess.py \
  --mindrecord_schema internlm2_alpaca \
  --input_glob {path}/alpaca_data.json \
  --output_file {path}/alpaca_processed/alpaca.mindrecord \
  --model_file {path}/tokenizer.model \
  --seq_length 2048

  # 参数说明
  mindrecord_schema:   描述自定义数据的schema类型
  input_glob:          输入下载后alpaca_data.json的文件路径
  output_file:         输出文件的保存路径
  model_file:          模型tokenizer.model文件路径
  seq_length:          输出数据的序列长度
```

### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。Base用于微调，Chat用于推理。

词表下载链接：[tokenizer.model](https://huggingface.co/internlm/internlm2-7b/blob/main/tokenizer.model)

|       模型名称        | MindSpore权重 |                        HuggingFace权重                       |
|:-----------------:|:-----------:|:--------------------------------------------------------------------:|
|   InternLM2-7B    |      \      |         [link](https://huggingface.co/internlm/internlm2-7b)         |
| InternLM2-chat-7B |      \      |       [link](https://huggingface.co/internlm/internlm2-chat-7b)      |
|   InternLM2-20B   |      \      | [link](https://huggingface.co/internlm/internlm2-20b)  |
|   InternLM2-chat-20B   |      \      | [link](https://huggingface.co/internlm/internlm2-chat-20b)  |

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

MindFormers提供`InternLM2-7B`的微调示例， 过程中使用alpaca数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

执行msrun启动脚本，进行8卡分布式微调

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --register_path research/internlm2 \
  --config research/internlm2/internlm2_7b/finetune_internlm2_7b.yaml \
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

多机多卡训练可以参考[多机多卡启动方式](https://gitee.com/mindspore/mindformers/blob/dev/README_CN.md#%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1)。

## 推理

`InternLM2-7B`的推理流程与`InternLM2-20B`相同。仅需替换配置文件。

### 单卡推理

以 `Internlm2-7B` 为例，即使用 `research/internlm2/internlm2_7b/predict_internlm2_7b.yaml` 配置文件。

### 单卡推理

修改配置文件 `research/internlm2/internlm2_7b/predict_internlm2_7b.yaml` ：

```yaml
processor:
  tokenizer:
    vocab_file: "./internlm2-7b/tokenizer.model"    # 指定tokenizer.model文件路径
```

执行如下推理命令。

```shell
python run_mindformer.py \
 --register_path research/internlm2 \
 --config research/internlm2/internlm2_7b/predict_internlm2_7b.yaml \
 --load_checkpoint /path/internlm2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel False \
 --run_mode predict \
 --predict_data '比较适合深度学习入门的书籍有'
# 比较适合深度学习入门的书籍有：
# 比较适合深度学习入门的书籍有《深度学习》、《机器学习》、《统计学习方法》、《统计自然语言处理》、《Python机器学习基础教程》...
```

### 多卡推理

`InternLM2`多卡推理暂不支持`is_dynamic=True`。本示例以 `InternLM2-20B` 2卡推理为例。

1. 修改配置文件 `research/internlm2/internlm2_20b/predict_internlm2_20b.yaml` ：

    ```yaml
    processor:
      tokenizer:
        vocab_file: "./internlm2-20b/tokenizer.model"    # 指定tokenizer.model文件路径
   ```

2. 执行如下推理命令，进行2卡推理

   ```shell
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/internlm2
    --config research/internlm2/internlm2_20b/predict_internlm2_20b.yaml \
    --load_checkpoint /path/internlm2_20b.ckpt \
    --auto_trans_ckpt True \
    --use_parallel True \
    --run_mode predict \
    --predict_data 比较适合深度学习入门的书籍有" 2
   # 比较适合深度学习入门的书籍有：
   # 比较适合深度学习入门的书籍有：\n1. 《深度学习》（Deep Learning）：这本书由深度学习领域的权威人士..
   ```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage