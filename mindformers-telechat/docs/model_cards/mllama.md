# Llama Vision 3.2

## 模型描述

Llama 3.2-Vision多模态大语言模型集合是预训练和指令调整的图像推理生成模型的集合，大小为11B和90B(文本+图像输入|文本输出)。Llama 3.2-Vision指令调整模型针对视觉识别、图像推理、字幕和回答有关的图像类问题进行了优化。在常见的行业基准上，它们的性能优于许多可用的开源和封闭式多模态模型。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                      |      Task       | Datasets | SeqLength | Performance  |  Phase  |
|:--------------------------------------------|:---------------:|:--------:|:---------:|:------------:|:-------:|
| [mllama_11b](./predict_mllama_11b.yaml)   | text_generation |    -     |   4096    | 1643 tokens/s | Predict |

## 模型文件

`Llama 3.2-Vision` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/mllama
       ├── __init__.py
       ├── mllama.py                  # 模型实现
       ├── mllama_config.py           # 模型配置项
       ├── mllama_tokenizer.py        # mllama tokenizer
       ├── mllama_processor.py        # mllama预处理
       └── mllama_transformer.py      # cross_attention层实现
   ```

2. 模型配置：

   ```bash
   configs/mllama
       ├── finetune_mllama2_11b.yaml         # 11b模型全量微调启动配置
       └── predict_mllama_11b.yaml           # 11b模型推理启动配置
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/models/mllama
       ├── data_process.py                # 数据集转换脚本
       ├── image_processing_mllama.py     # 图像数据处理脚本
       └── convert_weight.py              # 模型权重转换脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)
和[版本匹配关系](../../README.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持11b单机单卡推理，全参微调至少需要单机8卡，推荐使用单机8卡。

### 数据集及权重准备

#### 数据集下载

模型使用HuggingFace上的HuggingFaceM4/the_cauldron数据集中的ocrvqa作为微调数据集

| 数据集名称 |    适用模型     |   适用阶段   |                                下载链接                                |
|:------|:-----------:|:--------:|:------------------------------------------------------------------:|
| ocrvqa | mllama | Finetune | [Link](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/ocrvqa) |

通过以下脚本把下载好的arrow格式的数据集，转成json格式：

```shell
python mindformers\models\mllama\data_process.py --data_dir "input" --output_file "output"
```

运行完成，会在output目录生成以下数据：

```text
output
    ├── images               # 存放图片目录
    └── train_data.json      # 对话数据
```

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

| 模型名称         | MindSpore权重 |                        HuggingFace权重                         |
|:-------------|:-----------:|:------------------------------------------------------------:|
| Llama-3.2-11B-Vision-Instruct |      \      | [Link](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)  |
| Llama-3.2-11B-Vision |      \      | [Link](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)  |

> 注: 请自行申请huggingface上Llama-3.2-11B-Vision-Instruct使用权限，并安装transformers>=4.45版本

#### 模型权重转换

下载完成后，运行`mindformers/models/mllama/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --torch_ckpt_path TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
torch_ckpt_path:  下载HuggingFace权重的文件夹路径
mindspore_ckpt_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 微调

### 单机训练

以`llama3_2-vision-11b`为例，修改configs\mllama\finetune_mllama_11b.yaml配置文件，annotation_file替换为[数据处理](#数据集下载)好的train_data.json路径，vocab_file替换为对应的[词表](#模型权重下载)路径。

```yaml
train_dataset: &train_dataset
  data_loader:
    type: BaseMultiModalDataLoader
    annotation_file: "output/train_data.json" #本地生成的数据集文件
    shuffle: False
  ...
  tokenizer:
    pad_token: "<|finetune_right_pad_id|>"
    vocab_file: "path/tokenizer.model"        #替换本地tokenizer.model文件
    add_bos_token: True
    type: MllamaTokenizer
model:
  model_config:
  ...
   vision_model:
      arch:
        type: MllamaVisionModel
      model_config:
        type: MllamaVisionConfig
      ...
      image_size: &image_size 560      #instruct模型image_size为560，base模型image_size为448
      ...
```

执行msrun启动脚本，进行8卡分布式微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/mllama/finetune_mllama_11b.yaml \
 --load_checkpoint /{path}/mllama_11b.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8

# 参数说明
  config:  模型配置文件
  load_checkpoint:   模型路径
  auto_trans_ckpt:   是否开启模型自动切分
  use_parallel:   是否使用多卡并行训练
  run_mode:  运行模式
```

## 推理

MindFormers提供`Llama3_2-Vision`的推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。
配置`predict_mllama_11b.yaml`，修改vocab_file为对应的[词表](#模型权重下载)路径。推理输入默认添加bos字符，如果需要更改可在模型的yaml文件中修改add_bos_token选项。

```yaml
  tokenizer:
    add_bos_token: True
    add_eos_token: False
    vocab_file: "path/to/your/tokenizer.model"
    type: MllamaTokenizer
```

### 单卡推理

以`llama3_2-vision-11b`单卡推理为例。
调用`run_mindformer.py`公共接口，运行命令为：

```shell
python run_mindformer.py \
--config configs/mllama/predict_mllama_11b.yaml \
--run_mode predict \
--predict_data 'path/to/your_image.jpg' 'your text question' \
--modal_type image text \
--use_parallel False \
--auto_trans_ckpt False \
--load_checkpoint path/to/mllama_11b_instruct.ckpt

# 参数说明
  config:  模型配置文件
  run_mode:  运行模式
  predict_data: 模型推理输入, 第一个输入是图片路径, 第二个输入是文本
  modal_type:   模型推理输入对应模态, 图片路径对应'image', 文本对应'text'
  use_parallel: 是否使用多卡推理
  auto_trans_ckpt:   是否开启模型自动切分
  load_checkpoint:   模型路径
```

### 多卡推理

以`llama3_2-vision-11b`2卡推理为例。
调用`run_mindformer.py`公共接口，需要改动`predict_mllama_11b.yaml`中的配置，配置将`model_parallel`修改为需要使用的卡数，

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2  # 修改为需要使用的卡数
```

运行命令为：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/mllama/predict_mllama_11b.yaml \
--run_mode predict \
--predict_data 'path/to/your_image.jpg' 'your text question' \
--modal_type image text \
--use_parallel True \
--auto_trans_ckpt True \
--load_checkpoint path/to/mllama_11b_instruct.ckpt" 2

# 参数说明
bash scripts/msrun_launcher.sh COMMAND CKPT_PATH DEVICE_NUM
COMMAND: 执行命令
DEVICE_NUM:  使用卡数
```
