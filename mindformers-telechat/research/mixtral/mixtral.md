# Mixtral-8x7B

## 模型描述

Mixtral是MistralAI基于Mistral的更新版本，目前有4个版本：Mixtral-8x7B，Mixtral-8x7B-Instruct，Mixtral-8x22B，Mixtral-8x22B-Instruct。Mixtral模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。目前仓上提供的Mixtral模型是Mixtral-8x7B，其模型结构与Llama2-7b几乎一致，只是在Llama2-7b的基础上添加了MoE，并采用了分组查询注意力机制(GQA)。

[Mixtral of Experts](https://arxiv.org/abs/2401.04088)

``` text
@article{
  title={Mixtral of Experts},
  author={Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour and others},
  journal={arXiv preprint arXiv:2401.04088},
  year={2024}
}
```

## 模型文件

`Mixtral` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # mixtral网络层定义
       ├── llama_processor.py        # mixtral预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`research/mixtral`

   ```bash
   mixtral
    ├── pretrain_mixtral-8x7b.yaml        # 8x7b模型4k预训练启动配置
    ├── pretrain_mixtral-8x7b_32k.yaml    # 8x7b模型32k预训练启动配置
    ├── finetune_mixtral-8x7b.yaml        # 8x7b模型全参微调启动配置
    ├── predict_mixtral-8x7b.yaml         # 8x7b模型推理启动配置
    ├── convert_weight.py                 # 权重转换脚本（torch_to_ms）
    └── convert_reversed.py               # 权重反转脚本（ms_to_torch）
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

注：Atlas 800T A2芯片推理单机2卡即可推理.全参微调Atlas 800T A2芯片至少需要二机16卡。

### 数据及权重准备

#### 数据集下载

当前提供WikiText-103和alpaca_data数据集分别作为预训练和微调的数据集。

| 数据集名称             |               适用模型               |      适用阶段       |                                                 下载链接                                                  |
|:------------------|:--------------------------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|
| WikiText-103 | Mixtral-8x7b | pretrain | [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) |
| alpaca_data | Mixtral-8x7b | finetune / lora | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) |

1. 分词模型下载，参考模型权重下载的tokenizer.model

2. 使用以下预处理脚本生成mindrecord训练数据

``` bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/wiki4096.mindrecord
```

数据处理时候注意bos，eos，pad等特殊ids要和yaml配置中model_config里保持一致，默认bos_token_id=1, eos_token_id=2, pad_token_id=0, 如果有所修改，yaml中对应的配置也需要修改； 一般预训练的数据中不包含pad_token，此时建议pad_token_id设为-1。

- alpaca数据集原始格式样例：

```text
# alpaca examples:
    {
        "instruction": "Describe a time when you had to make a difficult decision.",
        "input": "",
        "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client\u2019s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team\u2019s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client\u2019s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
    },
    {
        "instruction": "Identify the odd one out.",
        "input": "Twitter, Instagram, Telegram",
        "output": "Telegram"
    },
```

- step 1. 执行`alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

``` bash
# 脚本路径：tools/dataset_preprocess/llama/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/alpaca_data.json \
--output_path /{path}/alpaca-data-conversation.json
```

```text
# 参数说明
data_path: 存放alpaca数据的路径
output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
{
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
```

- step 2. 执行`llama_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：tools/dataset_preprocess/llama/llama_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python llama_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/alpaca-fastchat4096.mindrecord
```

#### 模型权重下载

目前，MindFormers 未提供已经转换完成的预训练权重，用户需要从huggingface下载预训练权重，然后使用权重转换脚本将官方权重转换成Mindspore权重。

| 模型名称               |                                                    MindSpore权重                                                     |                         HuggingFace权重                          |
|:-------------------|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| Mixtral-8x7b  | / | [Link](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)  |
|tokenizer.model|[Link](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/tokenizer.model)|/|

#### 模型权重转换

1. torch权重转mindspore权重

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

> **注：此处的convert_weight.py指mindformers下的统一权重转换脚本**

```shell
python convert_weight.py \
--model mixtral \
--input_path {path}/torch_ckpt_dir \
--output_path {path}/ms_ckpt_path/mixtral.ckpt \
--dtype fp16
```

```text
# 参数说明
model: 指明待转换权重的模型。
input_path: huggingface权重保存目录路径,填写safetensors文件的保存路径即可。
output_path: ms权重保存文件路径，需指定到保存权重的名称。
dtype: 指定转换出的ms权重的数据类型。
```

2. mindspore权重转torch权重

在生成mindspore权重之后如需使用torch运行，可根据如下命令转换：

```shell
python convert_weight.py \
--model mixtral \
--reversed \
--input_path {path}/ms_ckpt_path/mixtral.ckpt \
--output_path {path}/torch_ckpt_dir/ \
--dtype fp16
```

```text
# 参数说明：
model: 指明待转换权重的模型。
reversed: 开启权重反转。
input_path: 待转换的mindspore权重（完整权重），需指定到.ckpt文件。
output_path: 转换后的torch权重存放路径，路径必须以'/'结尾。
dtype: 指定转换出的torch权重的数据类型。
```

若待转换的ms权重是分布式权重，可根据如下命令转换：

```shell
python convert_weight.py \
--model mixtral \
--reversed \
--input_path {path}/ms_ckpt_path/ \
--output_path {path}/torch_ckpt_dir/ \
--dtype fp16 \
--strategy_dir {path}/strategy_dir
```

```text
# 参数说明：
model: 指明待转换权重的模型。
reversed: 开启权重反转。
input_path: 待转换的mindspore权重（分布式权重）存放路径。
output_path: 转换后的torch权重存放路径，路径必须以'/'结尾。
dtype: 指定转换出的torch权重的数据类型。
strategy_dir: 待转换mindspore权重的strategy文件路径
```

#### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix mixtral_
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 预训练

### 多机训练

- step 1. 在模型对应的配置文件`research/mixtral/pretrain_mixtral-8x7b.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练) 通过配置中的`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

- step 2. 根据服务器节点数等信息，修改相应的配置。

``` bash
# 以mixtral-8x7b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../research/mixtral/pretrain_mixtral-8x7b.yaml
parallel_config:
  data_parallel: 8
  model_parallel: 1
  expert_parallel: 8
  pipeline_stage: 2
  use_seq_parallel: False
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

> **注：多机多卡并行配置中data_parallel\*model_parallel\*pipeline_stage == 总卡数，且expert_parallel不能超过data_parallel。此模型暂不支持配置`context_parallel`，因此暂不支持长序列。**

- step 3. 调大`moe_config`中的专家容量因子`capacity_factor`(非必要步骤)

``` bash
# capacity_factor默认值为1.1，调大capacity_factor(建议值2.0-4.0)可提高训练精度，但会带来一定的性能损失
# moe
moe_config:
  expert_num: 8
  capacity_factor: 2.0
  aux_loss_factor: 0.05
  num_experts_chosen: 2
  routing_policy: "TopkRouterV2"
  enable_sdrop: True
  router_dense_type: "float32"
```

- step 4. 执行运行脚本。

在多机上同时拉起任务，将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，具体可参考[ms_run快速使用](https://gitee.com/mindspore/mindformers#%E5%9B%9B%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8)

```shell
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
cd mindformers/research
bash ../scripts/msrun_launcher.sh "../run_mindformer.py \
 --config research/mixtral/pretrain_mixtral-8x7b.yaml \
 --run_mode train \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
cd mindformers/research
bash ../scripts/msrun_launcher.sh "../run_mindformer.py \
 --config research/mixtral/pretrain_mixtral-8x7b.yaml \
 --run_mode train \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 1 output/msrun_log False 300

# 参数说明，启动命令中的参数会覆盖yaml文件中的相同参数
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

## 微调

### 全参微调

- step 1. 将`../research/mixtral/finetune_mixtral-8x7b.yaml`中训练数据集路径改为微调数据集路径。

```bash
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 默认开启`moe_config`中的`enable_sdrop`选项，修改微调时学习率, 优化器参数，`seq_length`, 与预训练不同，微调配置如下：

```bash
# moe
moe_config:
  expert_num: 8
  capacity_factor: 1.1
  aux_loss_factor: 0.05
  num_experts_chosen: 2
  routing_policy: "TopkRouterV2"
  enable_sdrop: True
  router_dense_type: "float32"

# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.95
  eps: 1.e-8
  learning_rate: 3.e-4

# lr schedule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 4096
```

- step 3. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。若无共享磁盘，跨机需先手动切分权重，详细教程请参考特性文档模型[权重切分与合并](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)

- step 4. 启动微调任务，以两机16卡为例进行微调，命令如下：

```shell
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
cd mindformers/research
bash ../scripts/msrun_launcher.sh "../run_mindformer.py \
 --config research/mixtral/finetune_mixtral-8x7b.yaml \
 --load_checkpoint model_dir \
 --run_mode finetune \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
cd mindformers/research
bash ../scripts/msrun_launcher.sh "../run_mindformer.py \
 --config research/mixtral/finetune_mixtral-8x7b.yaml \
 --load_checkpoint model_dir \
 --run_mode finetune \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 1 output/msrun_log False 300

 # 参数说明，启动命令中的参数会覆盖yaml文件中的相同参数
config: 配置文件路径
load_checkpoint: 切分好的分布式权重文件夹路径，权重按照model_dir/rank_x/xxx.ckpt格式，文件夹路径填写为model_dir
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

## 推理

Mixtal-8x7b使用单机多卡推理，使能 gmm 算子优化需要对权重进行转换，转换命令如下：

```shell
# pytorch权重 转换 ms_gmm权重
python convert_weight.py --use_gmm=True --dtype='fp16' --torch_ckpt_idr=/path/to/mixtral_torch_ckpt_dir --mindspore_ckpt_path=/path/to/new_mixtral.ckpt

# pytorch权重 转换 mindspore权重
python convert_weight.py --use_gmm=False --dtype='fp16' --torch_ckpt_idr=/path/to/mixtral_torch_ckpt_dir --mindspore_ckpt_path=/path/to/new_mixtral.ckpt

# mindspore权重 转换 ms_gmm权重
python convert_weight.py --use_gmm=True --dtype='fp16' --pre_ckpt_path=/path/to/mixtral.ckpt --mindspore_ckpt_path=/path/to/new_mixtral.ckpt

# 参数说明
use_gmm: 是否进行gmm转换
dtype: 数据类型，可选择: 'fp16'，'fp32'，'bf16', 默认值为 'fp16'
torch_ckpt_idr: 需要转换的torch权重的目录
pre_ckpt_path: 需要转换的ms权重的路径
mindspore_ckpt_path: 转换后gmm的权重的存储路径
 ```

 模型权重转换完成后，可以用如下命令进行单机多卡推理：

```shell
cd mindformers/research
bash ../scripts/msrun_launcher.sh "../run_mindformer.py \
 --config research/mixtral/predict_mixtral-8x7b.yaml \
 --run_mode predict \
 --use_parallel True \
 --load_checkpoint model_dir \
 --auto_trans_ckpt True \
 --predict_data \"I love Beijing, because\"" \
 8 8118 output/msrun_log False 300

# 输出推理结果：I love Beijing, because it is a city of contrats. It is a city of the old and the new, the traditional ... ... the future.

# 参数说明，启动命令中的参数会覆盖yaml文件中的相同参数
config: 配置文件路径
run_mode: 运行模式，推理时设置为predict
load_checkpoint: 权重路径
auto_trans_ckpt: 是否开启自动权重转换
predict_data：需要推理的问题
 ```

> **注：当前mindformers的MoE属于DropMoE（有token丢弃），推理需要DroplessMoE。当前推理仅支持`use_past=False`,开启`moe_config`中的`enable_sdrop`选项可以在BS=1场景下用DropMoE达到DroplessMoE的效果，但在多Batch场景暂不支持，DroplessMoE待后续训推一体支持。**
