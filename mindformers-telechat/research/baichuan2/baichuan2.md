# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。
目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                         |      Task       | SeqLength | Datasets |   Performance   |  Phase   |
|:-----------------------------------------------|:---------------:|:---------:|:--------:|:---------------:|:--------:|
| [baichuan2_7b](./finetune_baichuan2_7b.yaml)   | text_generation |   4096    |  belle   | 3164 tokens/s/p | Finetune |
| [baichuan2_13b](./finetune_baichuan2_13b.yaml) | text_generation |   4096    |  belle   | 1465 tokens/s/p | Finetune |
| [baichuan2_7b](./predict_baichuan2_7b.yaml)    | text_generation |   4096    |    /     |  521 tokens/s   | Predict  |
| [baichuan2_13b](./predict_baichuan2_13b.yaml)  | text_generation |   4096    |    /     |  224 tokens/s   | Predict  |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                         |      Task       | SeqLength | Datasets |   Performance   |  Phase   |
|:-----------------------------------------------|:---------------:|:---------:|:--------:|:---------------:|:--------:|
| [baichuan2_13b](./finetune_baichuan2_13b.yaml) | text_generation |   4096    |  belle   | 1640 tokens/s/p | Finetune |

## 模型文件

`Baichuan2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/baichuan2
       ├── baichuan2_tokenizer.py    # tokenizer
       ├── baichuan2_7b.py           # 7B模型实现
       └── baichuan2_13b.py          # 13B模型实现
   ```

2. 模型配置：

   ```text
   research/baichuan2
       ├── finetune_baichuan2_7b.yaml               # 7B全参微调配置
       ├── finetune_baichuan2_13b.yaml              # 13B全参微调配置
       ├── finetune_baichuan2_7b_lora.yaml          # 7B-lora微调配置
       ├── finetune_baichuan2_13b_lora.yaml         # 13B-lora微调配置
       ├── predict_baichuan2_7b.yaml                # 7B单卡推理配置
       └── predict_baichuan2_13b.yaml               # 13B单卡推理配置
   ```

3. 数据预处理和任务启动脚本：

   ```text
   research/baichuan2
       ├── convert_weight.py       # hf->mf权重转换
       ├── convert_reversed.py     # mf->hf权重转换
       ├── belle_preprocess.py     # belle数据集预处理脚本
       ├── run_baichuan2.py        # baichuan2高阶接口使用脚本
       └── run_baichuan2_chat.py   # baichuan2 chat推理使用脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持Baichuan2-7b和Baichuan2-13b进行单机8卡全参微调和LoRA微调，以及单机单卡推理。

### 数据及权重准备

#### 数据集下载

当前提供belle_chat_ramdon数据集的预处理和微调样例，用于对Baichuan2-7B-Base，Baichuan2-13B-Base模型进行微调。

| 数据集名称             |               适用模型               |      适用阶段       |                                                 下载链接                                                  |
|:------------------|:--------------------------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|
| belle_chat_ramdon | Baichuan2-7B <br/> Baichuan2-13B | finetune / lora | [Link](https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/data/belle_chat_ramdon_10k.json) |

下载数据集后，需要执行`belle_preprocess.py`脚本进行数据预处理，将原始数据转换为mindrecord格式。`tokenizer.model`可点击[链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)进行下载。

```shell
cd research/baichuan2
python belle_preprocess.py \
 --input_glob /path/to/belle_chat_ramdon_10k.json \
 --model_file /path/to/tokenizer.model \
 --output_file /path/to/belle_chat_ramdon_10k_4096.mindrecord \
 --seq_length 4096

# 参数说明
input_glob: 原始数据集文件路径
model_file: 词表文件路径
output_file: 输出数据集文件路径
seq_length: 生成数据集的序列长度
```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

也可选择从HuggingFace下载所有工程文件后进行[模型权重转换](#模型权重转换)使用。

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

| 模型名称               |                                                    MindSpore权重                                                     |                         HuggingFace权重                          |
|:-------------------|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| Baichuan2-7B-Base  | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Base.ckpt)  | [Link](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)  |
| Baichuan2-7B-Chat  | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt)  | [Link](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |
| Baichuan2-13B-Base | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_13B_Base.ckpt) | [Link](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) |
| Baichuan2-13B-Chat | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt) | [Link](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) |

#### 模型权重转换

进行权重转换需要安装以下依赖包。

```shell
pip install torch transformers protobuf wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py \
 --input_path path/to/baichuan2/checkpoints/ \
 --output_path path/to/baichuan2.ckpt \
 --model baichuan2 \
 --dtype fp16

# 参数说明
input_path: HuggingFace权重保存目录路径，该目录下存放权重和相关的一些配置文件等，建议直接克隆HuggingFace，保存全部文件到本地
output_path: 转换后的MindSpore权重文件保存路径
model: 模型名字
dtype: 权重的精度
```

## 微调

MindFormers提供Baichuan2全参微调和LoRA微调的示例。

### 全参微调

MindFormers提供了默认微调配置`finetune_baichuan2_7b.yaml`和`finetune_baichuan2_13b.yaml`，
默认使用数据集[belle_chat_ramdon](#数据集下载)，并在配置文件中设置`seq_length=4096`。

#### 单机训练

以Baichuan2-7B为例，Baichuan2-13B只需要修改配置文件和权重即可。

使用msrun方式拉起分布式训练，默认使用8卡进行训练，如果需要修改使用卡数参考[使用指南](../../README.md#三使用指南)进行配置。

```shell
cd research/baichuan2
bash ../../scripts/msrun_launcher.sh \
 "run_baichuan2.py \
 --config finetune_baichuan2_7b.yaml \
 --load_checkpoint path/to/baichuan2_7b_base.ckpt \
 --auto_trans_ckpt True \
 --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord"

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
train_data: 训练数据集文件夹路径或mindrecord文件路径
```

#### 多机训练

以Baichuan2-7B进行2机8卡训练为例，Baichuan2-13B只需要修改配置文件和权重即可。

1. 修改`finetune_baichuan2_7b.yaml`

    ```yaml
    parallel_config:
      data_parallel: 8
      model_parallel: 1
      pipeline_stage: 2
      use_seq_parallel: False
      micro_batch_num: 8
    ```

2. 设置环境变量

    ```shell
    export MS_MEMORY_POOL_RECYCLE=1
    export GE_NOT_CUT=1
    ```

3. 使用msrun脚本训练

   多机训练需要分别在不同节点执行命令，以下为2机8卡训练过程，参数说明以及使用更多节点参考[使用指南](../../README.md#三使用指南)多机多卡部分进行配置。

   > 注：如果各节点间使用共享存储存放工程文件，则可以使用**支持权重自动转换**功能，否则需要修改配置文件`auto_trans_ckpt=False`或在运行命令时设置`--auto_trans_ckpt False`，
   此时，预训练权重可以使用[权重转换工具](../../docs/feature_cards/Transform_Ckpt.md)转换得到。

   针对多机的场景，建议用户配置HCCL_BUFFSIZE环境变量。集合通信网络中，每一个HCCL通信域都会占用HCCL_BUFFSIZE大小的缓存区，若业务的模型数据量较小，但通信数据量较大，则可通过此环境变量增大HCCL通信域占用的缓存区大小，提升数据通信效率。

   ```shell
   export HCCL_BUFFSIZE=2
   ```

- 在节点0执行如下命令，其中192.168.1.1需要改为节点0的实际ip，将节点0作为主节点，2机共16卡且每个节点8卡。

  ```shell
  # 以使用共享盘为例
  bash ../../scripts/msrun_launcher.sh \
   "run_baichuan2.py \
   --config finetune_baichuan2_7b.yaml \
   --load_checkpoint path/to/baichuan2_7b_base.ckpt \
   --auto_trans_ckpt True \
   --use_parallel True \
   --run_mode finetune \
   --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300
  ```

- 在节点1执行如下命令，其中192.168.1.1需要改为节点0的实际ip。

  ```shell
  bash ../../scripts/msrun_launcher.sh \
   "run_baichuan2.py \
   --config finetune_baichuan2_7b.yaml \
   --load_checkpoint path/to/baichuan2_7b_base.ckpt \
   --auto_trans_ckpt True \
   --use_parallel True \
   --run_mode finetune \
   --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 300
  ```

  > 注：训练过程中出现问题，可以参考[**常见问题**](#常见问题)进行解决。

### LoRA微调

#### 单机训练

LoRA微调执行命令与[全参微调](#全参微调)相同，修改`--config`为`finetune_baichuan2_7b_lora.yaml`或`finetune_baichuan2_13b_lora.yaml`即可。

如果仅需要单独保存Lora权重，修改配置文件中相关参数：

```yaml
callbacks:
  - type: CheckpointMonitor
    save_trainable_params: True
```

> 注：微调完成后，可使用lora权重合并工具将Lora权重和Base权重合并，具体说明可参考[lora权重合并教程](../../docs/feature_cards/Transform_Lorackpt.md)。

## 推理

MindFormers提供`Baichuan2`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

使用推理功能时，推荐使用`Baichuan2-7B-Chat`和`Baichuan2-13B-Chat`权重，默认设置`seq_length=4096`，模型权重以及tokenizer文件可参考[模型权重下载](#模型权重下载)。

```shell
# 脚本使用
bash scripts/examples/baichuan2/run_baichuan2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH TOKENIZER INPUT_DATA DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
TOKENIZER:   模型tokenizer文件路径
INPUT_DATA:  输入模型预测数据
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

> 注：如果输入数据中前后`"`需要添加转义字符`'\'`。例如，推理`"I love you."`，则`INPUT_DATA`应该为`\"I love you.\"`。

### 单卡推理

```shell
# baichuan2 7b
bash scripts/examples/baichuan2/run_baichuan2_predict.sh single \
 research/baichuan2/predict_baichuan2_7b.yaml \
 path/to/baichuan2_7b_chat.ckpt \
 path/to/tokenizer.model \
 "你是谁？"
# 输出推理结果：我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问

# baichuan2 13b
bash scripts/examples/baichuan2/run_baichuan2_predict.sh single \
 research/baichuan2/predict_baichuan2_13b.yaml \
 path/to/baichuan2_13b_chat.ckpt \
 path/to/tokenizer.model \
 "你是谁？"
# 输出推理结果：我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问
```

### 多卡推理

`Baichuan2`多卡推理暂不支持`is_dynamic=True`，示例中使用2卡进行推理，多卡推理输出推理结果与单卡推理相同。

```shell
# baichuan2 7b
bash scripts/examples/baichuan2/run_baichuan2_predict.sh parallel \
 research/baichuan2/predict_baichuan2_7b.yaml \
 path/to/baichuan2_7b_chat.ckpt \
 path/to/tokenizer.model \
 "你好。" 2

# baichuan2 13b
bash scripts/examples/baichuan2/run_baichuan2_predict.sh parallel \
 research/baichuan2/predict_baichuan2_13b.yaml \
 path/to/baichuan2_13b_chat.ckpt \
 path/to/tokenizer.model \
 "你是谁？" 2
```

## 常见问题

### 显存不足

如出现显存不足问题，可适当减少`data_paralell`并增大`model_parallel`，如`data_parallel=4`、`model_paralllel=2`
