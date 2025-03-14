# LLaVA-NeXT

## 模型描述

LLaVA NeXT是一个端到端训练的大型多模态模型，连接视觉编码器和大语言模型，以实现通用视觉和语言理解，通过在 GPT 生成的多模式指令跟踪数据上微调 LLaMA系列模型进行训练。LLaVA NeXT相比LLaVA1.5使用了更高分辨率的图片数据并引入相关分辨率数据处理逻辑。它是一种基于 Transformer 架构的自回归语言模型。MF暂支持LLaVA NeXT Video 7b模型

```text
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}
```

## 模型文件

`Llava-NEXT` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/llava_next/
       ├── llava_next.py                        # 模型实现
       ├── llava_next_vision_tower.py           # 视觉编码器实现
       ├── llava_next_multi_modal_processor.py  # 视觉输入处理器实现
       ├── prompt_processor.py                  # 提示处理器实现
       ├── llava_next_tokenizer.py              # 模型分词器实现
       ├── llava_anyres_process.py              # 高分辨率图片处理实现
       └── llava_next_config.py                 # 模型配置
   ```

2. 模型配置：

   ```text
   research/llava_next/
       └──llava_next_video_v1_7b
           ├── finetune_llava_next_video_v1_7b_stage2.yaml   # 7B阶段2配置(纯单图)
           ├── finetune_llava_next_video_v1_7b_stage3.yaml   # 7B阶段3配置(纯视频)
           ├── predict_llava_next_video_v1_7b_stage2.yaml   # 7B图片推理配置
           └── predict_llava_next_video_v1_7b_stage3.yaml  # 7B视频推理配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

|        模型         |     硬件      | 训练 |
| :-----------------: | :-----------: | :--: |
| LLaVA-NeXT-Video-7B | Atlas 800T A2 | 八卡 |

### 数据及权重准备

#### 数据集下载

##### 视频

MindFormers提供**Video-178K**中的[0_30_s_academic_oe_v0_1_qa_processed.json](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.json)作为demo[微调](#微调)**视频**的数据集。

| 数据集名称                                |      适用模型       |     适用阶段     |                           下载链接                           |
| :---------------------------------------- | :-----------------: | :--------------: | :----------------------------------------------------------: |
| 0_30_s_academic_oe_v0_1_qa_processed.json | LlaVA-NeXT-Video-7B | Finetune stage 3 | [Link](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/0_30_s_academic_v0_1) |

##### 图片

MindFormers提供**LLaVA-Instruct-150K**中的[llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)中的coco子数据作为demo[微调](#微调)**图片**的数据集。若链接跳转失败，可手动复制粘贴 http://images.cocodataset.org/zips/train2017.zip 至地址栏访问下载。

| 数据集名称 |      适用模型       |     适用阶段     |                         下载链接                         |
| :--------- | :-----------------: | :--------------: | :------------------------------------------------------: |
| coco       | LlaVA-NeXT-Video-7B | Finetune stage 2 | [Link](http://images.cocodataset.org/zips/train2017.zip) |

数据处理：

将连接中的压缩文件与json下载下来，并将压缩文件解压。

将该json文件与解压完成的视频或者图片文件夹放在**同一个目录**下面，如下面的数据位置路径所示。

```text
# 数据位置路径
视频文件夹
    ├── 0_30_s_academic_oe_v0_1_qa_processed.json   # 文本数据
    └── academic_source                             # 视频数据路径
图片文件夹
    ├── llava_v1_5_mix665k.json                     # 文本数据
    └── coco
         └── train2017                              # 图片数据路径

# 数据处理代码路径
research/llava_next
    ├── data_process.py   # 数据处理路径
    └── data_process.yaml  # 数据路径
```

```yaml
# >> data_process.yaml
# 将数据集放入data_process.yaml里面，结构如下
datasets:
  - json_path: XXX.json
    sampling_strategy: "first:XX%"
  - json_path: XXX.json
    sampling_strategy: "all"
  - json_path: XXX.json
    sampling_strategy: "end:XX%"
json_path: 数据集路径
sampling_strategy: 采样取值，first:从前面起取多少数据，end:从后开始取百分之多少数据；random: 随机取多少值；all:全部取值。
```

随后运行下面代码，获得数据集处理json文件。

```shell
python data_process.py --data_yaml data_process.yaml --output_file OUTPUTFILE
```

### 权重准备

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表相关下载链接：[tokenizer.model](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/blob/main/tokenizer.model)，[tokenizer_config.json](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/blob/main/tokenizer_config.json), [generation_config.json](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/blob/main/generation_config.json), [added_tokens.json](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/blob/main/added_tokens.json)

请安装`pip install tokenizer>=0.20.3`

| 模型名称            | MindSpore权重 |             HuggingFace权重             |
| :------------------ | :-----------: | :-------------------------------------: |
| LLaVA-NeXT-Video-7B |       \       | [Link](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重(`transformers>=4.45.2`)。

**完整模型**：使用上述权重直接下载后进行权重转换。

**单模型组合**：自定义视觉模型和语言模型，在huggingface上下载后进行权重转换，转换后将两个模型都放入一个文件夹中。

```shell
# 完整模型
python convert_weight.py --model llava_next --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 单模型组合训练
# 单视频模型
python convert_weight.py --model llava_next --input_path VISION_TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --from_vision True
# 单语言模型
python convert_weight.py --model llava_next --input_path LANG_TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --from_language True

# 参数说明
model:         模型名称
input_path:    下载HuggingFace权重的文件夹路径
output_path:   转换后的MindSpore权重文件保存路径
from_vison:    单视觉模型转换开关
from_language: 单语言模型转换开关
```

## 推理

MindFormers提供`LlaVA-NeXT`的推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。
配置`predict_llava_next_video_v1_7b_stage3.yaml`，将词表相关所在文件夹写入vocab_files, 文件夹下放tokenizer的文件。

```yaml
  tokenizer:
    add_bos_token: True
    add_eos_token: False
    vocab_file: "path/to/your/tokenizer_files"   # 给文件夹即可，文件夹下放下载的词表相关的文件
    image_tag: "<image>"
    type: LlavaNextTokenizer
```

### 单卡推理

以`LLaVA-NeXT-Video-7b`单卡推理为例。
调用`run_mindformer.py`公共接口
`video`: 使用`predict_llava_next_video_v1_7b_stage3.yaml`，`modal_type`使用`video text`
`image`: 使用`predict_llava_next_video_v1_7b_stage2.yaml`， `modal_type`使用`image text`
运行命令如下

```shell
python run_mindformer.py \
--config research/llava_next/llava_next_video_v1_7b/predict_llava_next_video_v1_7b_stage3.yaml \
--register_path research/llava_next \
--run_mode predict \
--predict_data 'path\to\your_video.mp4' 'your text question' \
--modal_type video text \
--use_parallel False \
--auto_trans_ckpt False \
--load_checkpoint path/to/llava_next_video_7b.ckpt
```

### 多卡推理

以`LLaVA-NeXT-Video-7b`2卡推理为例。
调用`run_mindformer.py`公共接口，需要改动`predict_llava_next_video_v1_7b_stage3.yaml`中的配置，配置将`model_parallel`修改为需要使用的卡数，

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2  # 修改为需要使用的卡数
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 1
```

运行命令为：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config research/llava_next/llava_next_video_v1_7b/predict_llava_next_video_v1_7b_stage3.yaml \
--register_path research/llava_next \
--run_mode predict \
--predict_data 'path\to\your_video.mp4' 'your text question' \
--modal_type video text \
--use_parallel True \
--auto_trans_ckpt True \
--load_checkpoint path/to/llava_next_video_7b.ckpt" 2
```

## 微调

MindFormers提供`LLaVA-NeXT-Video-7b`的微调示例，过程中使用`alpaca`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

以`LLaVA-NeXT-Video-7b`为例，将处理好到数据集，tokenizer所在文件夹放入`finetune_llava_next_video_v1_7b_stage3.yaml`中。

数据集，使用上述数据处理流程得到：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: BaseMultiModalDataLoader
    annotation_file: "你的json数据集路径"
    shuffle: False
  num_parallel_workers: 8
  python_multiprocessing: True
```

tokenizer：给出模型使用的tokenizer的配置，`bos`，`eos`,`pad`,`tokenizer_type` 根据语言模型使用相应修改。

```yaml
  tokenizer:
    add_bos_token: True
    add_eos_token: False
    vocab_file: "/path/to/tokenizer_files"  # 给文件夹即可，文件夹下放下载的词表相关的文件
    image_tag: "<image>"
    type: LlavaNextTokenizer
```

执行msrun启动脚本，进行8卡分布式训练。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config research/llava_next/llava_next_video_v1_7b/finetune_llava_next_video_v1_7b_stage3.yaml \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --register_path research/llava_next \
 --run_mode finetune" 8
```
