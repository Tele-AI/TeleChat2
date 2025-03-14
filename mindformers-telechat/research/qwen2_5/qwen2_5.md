# 通义千问

## 模型描述

Qwen2.5是Qwen系列的新的大型语言模型。Qwen2.5发布了许多基本语言模型和指令调整的语言模型，参数范围从5亿到720亿，包括专家混合模型。
与最先进的开源语言模型（包括之前发布的Qwen1.5）相比，Qwen2.5总体上超越了大多数开源模型，并在一系列针对语言理解，语言生成，多语言能力，编码，数学，推理等的基准测试中表现出对专有模型的竞争力。

```text
@article{qwen2_5,
  title={qwen2_5 Technical Report},
  year={2024}
}
```

## 模型性能

| Config                                             |      Task       |     Datasets      | SeqLength |  Phase   |             Performance             |
|:---------------------------------------------------|:---------------:|:-----------------:|:---------:|:--------:|:-----------------------------------:|
| [qwen2_5-7b](./predict_qwen2_5_7b_instruct.yaml)   | text_generation |  -  |   32768    | Predict  | - tokens/s(mindie 16 batch_size 单卡) |
| [qwen2_5-14b](./predict_qwen2_5_14b_instruct.yaml) | text_generation |  -  |   32768    | Predict  | - tokens/s(mindie 16 batch_size 单卡) |
| [qwen2_5-32b](./predict_qwen2_5_32b_instruct.yaml) | text_generation |  -  |   32768    | Predict  | - tokens/s(mindie 16 batch_size 双卡) |
| [qwen2_5-72b](./predict_qwen2_5_72b_instruct.yaml) | text_generation |  -  |   32768    | Predict  | - tokens/s(mindie 16 batch_size 四卡) |

## 模型文件

`qwen2_5` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/qwen2_5
     └── qwen2_5_tokenizer.py                        # tokenizer
   ```

2. 模型配置：

   ```text
   research/qwen2_5
     ├── predict_qwen2_5_7b_instruct.yaml             # 7B 在线推理启动配置
     ├── predict_qwen2_5_14b_instruct.yaml            # 14B 在线推理启动配置
     ├── predict_qwen2_5_32b_instruct.yaml            # 32B 在线推理启动配置
     ├── predict_qwen2_5_72b_instruct.yaml            # 72B 在线推理启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   research/qwen2_5
     ├── convert_weight.py                         # 权重转换脚本
     └── run_qwen2_5.py                              # qwen2_5多轮对话脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供`alpaca`作为[微调](#微调)数据集。

数据预处理中所用的`vocab.json`和`merges.txt`可以参考[模型权重下载](#模型权重下载)进行下载。

- **alpaca 数据预处理**

  1. 执行`research/qwen2/alpaca_converter.py`，将原始数据集转换为指定格式。(静态shape)
  2. 执行`research/qwen2/alpaca_converter_json.py`，将原始数据集转换为指定格式。(动态shape)

  ```shell
  python alpaca_converter.py \
   --data_path path/alpaca_data.json \
   --output_path /path/alpaca-data-messages.json

  # 参数说明
  data_path:   输入下载的文件路径
  output_path: 输出文件的保存路径
  ```

  执行`research/qwen2/qwen2_preprocess.py`文件，进行数据预处理和Mindrecord数据生成。

  ```shell
  python qwen2_preprocess.py \
   --dataset_type 'qa' \
   --input_glob /path/alpaca-data-messages.json \
   --vocab_file /path/vocab.json \
   --merges_file /path/merges.txt \
   --seq_length 32768 \
   --output_file /path/alpaca-messages.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   转换后的alpaca的文件路径
  vocab_file:   vocab.json文件路径
  merges_file:  merges.txt文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

#### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，`vocab.json`和`merges.txt`文件也在链接中下载。

词表下载链接：[vocab.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/vocab.json)和[merges.txt](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/merges.txt)

| 模型名称                 |                Base权重（建议训练和微调使用）                |                    Instruct权重（建议推理使用）                    |
|:---------------------|:-----------------------------------------------:|:--------------------------------------------------------:|
| qwen2_5-7b-Instruct  | [Link](https://huggingface.co/Qwen/Qwen2.5-7B)  | [Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)  |
| qwen2_5-14b-Instruct | [Link](https://huggingface.co/Qwen/Qwen2.5-14B) | [Link](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| qwen2_5-32b-Instruct | [Link](https://huggingface.co/Qwen/Qwen2.5-32B) | [Link](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| qwen2_5-72b-Instruct | [Link](https://huggingface.co/Qwen/Qwen2.5-72B) | [Link](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model qwen2_5 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
is_lora:     转换的权重是否是lora
align_rank:  lora配置中rank的值是否对齐
```

#### lora模型权重转换

注：align_rank参数控制lora配置文件参数 'r' 的是否对齐16。Atlas 300V Pro型号机器需要开启对齐。

```shell
python convert_weight.py --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16 --is_lora True --align_rank True
```

- **[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)**

  从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

  通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

  以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

#### 模型权重qkv_concat转换

- Qwen2.5系列默认打开qkv_concat参数，使用的权重需经过qkv_concat转换

```shell
python convert_weight.py  --qkv_concat True --model qwen2_5  --input_path {path}/MS_CKPT_NAME  --output_path {outputPath}/MS_CKPT_QKV_NAME

# 参数说明
qkv_concat:             是否开启qkv_concat,默认为false
model:                  调用哪个模型的脚本进行权重转换
pre_ckpt_path:          转化后的MindSpore权重文件保存路径,单卡权重指向文件,多卡权重指向文件夹
mindspore_ckpt_path:    qkv_concat转换后权重文件保存路径,单卡权重指向文件,多卡权重指向文件夹
```

## 微调

注意事项：

1. 当前支持模型已提供推理相关配置文件，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen2_5`目录下，或者先将`research/qwen2_5`目录所在路径加入到`PYTHONPATH`环境变量中。

以``qwen2_5-7b` 8卡微调为例，执行如下命令进行微调，微调前请参考[权重转换](../../docs/feature_cards/Transform_Ckpt.md)切分权重。

1. 主要参数配置参考:

- 基本配置：

  ```yaml
   load_checkpoint: './path/qwen2_5_7b.ckpt' # 权重转换后的文件
   train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "./path/alpaca-data.mindrecord" # 实际微调数据集
      shuffle: True
   # parallel config
   parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 1
    use_seq_parallel: True
    micro_batch_num: 1
    vocab_emb_dp: False
    gradient_aggregation_group: 4
   micro_batch_interleave_num: 2
   # processor config
   processor:
    return_tensors: ms
    tokenizer:
      model_max_length: 32768
      vocab_file: "./path/vocab.json" # 参考qwen2_5-7b官网下载的词表
      merges_file: "./path/merges.txt" # # 参考qwen2_5-7b官网下载的merge文件
  ```

- 动态shape配置：

```yaml
# model config
model:
  model_config:
    is_dynamic: True
# dataset
train_dataset: &train_dataset
  data_loader:
    type: SFTDataLoader
    dataset_dir: "./path/alpaca-data-json.json"
    tokenizer:
      unk_token: '<|endoftext|>'
      eos_token: '<|endoftext|>'
      pad_token: '<|endoftext|>'
      type: Qwen2Tokenizer
      vocab_file: "./path/vocab.json"
      merges_file: "./path/merges.txt"
    max_length: 32768
    file_format: json
    dataset_name: multi-round-chat-dyn-alpaca
    shuffle: False
    map_function_kwargs: {"user_prompt":"system\nYou are a helpful assistant.", "user_prompt_role":"user\n", "assistant_prompt_role":"assistant\n"}
    num_samples: 20000
  pad_token_id: 151643
  divisor: 4
  remainder: 1
  input_columns: ["input_ids", "labels"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  dynamic_batch: True
```

2. 配置并行加速

   若模型的yaml中有类似如下的配置

   ```yaml
   context:
     ascend_config:
       parallel_speed_up_json_path: "/path/to/parallel_speed_up.json"  # Replace with a real path when needed
   ```

   这是在使用`parallel_speed_up`文件（须是`json`格式）去配置一些并行加速特性，以获得一些性能上的提升。实际使用时，请把`json`文件的路径修改为实际值。

   > `parallel_speed_up`文件中各配置项的含义详见[parallel_speed_up说明](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html#:~:text=parallel_speed_up_json_path)。

   例如，`finetune_qwen2.5_72B_32K.yaml`中使用了`parallel_speed_up_72B_32K.json`，其中配置了`"matmul_grad_comm_overlap": true`。

3. 启动微调:

   ```shell
   cd research/qwen2_5
   bash ../../scripts/msrun_launcher.sh "run_qwen2_5.py \
    --config finetune_qwen2_5_7b.yaml \
    --run_mode finetune \
    --train_data ./path/alpaca-data.mindrecord "
   ```

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

注意事项：

1. 当前支持模型已提供推理相关配置文件，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen2_5`目录下，或者先将`research/qwen2_5`目录所在路径加入到`PYTHONPATH`环境变量中。

### 基于高阶接口的推理

#### 多卡推理

以`qwen2_5_72b`4卡推理为例，执行如下命令进行推理, 推理前先参考[权重转换](../../docs/feature_cards/Transform_Ckpt.md)切分权重。

1. 主要参数配置参考：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: False
     gradient_aggregation_group: 4
   ```

2. 启动多卡推理：

   ```shell
   cd research/qwen2_5
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash ../../scripts/msrun_launcher.sh "run_qwen2_5.py \
    --config predict_qwen2_5_72b_instruct.yaml \
    --load_checkpoint /path/model_dir \
    --vocab_file /path/vocab.json \
    --merges_file /path/merges.txt \
    --run_mode predict \
    --use_parallel True \
    --auto_trans_ckpt False \
    --predict_data 帮助我制定一份去上海的旅游攻略" 4

   # 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
   ```
