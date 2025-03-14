# 一招金融大模型(yizhao)

## 模型描述

一招（YiZhao)
是招商银行联合华为、智谱AI，在全国产的设备与训练框架下训练得到的金融领域大语言模型系列。我们首先在大规模高质量数据上进行预训练，再通过高质量金融数据进行后期微调（SFT），最后通过直接偏好优化（DPO）进一步优化模型，最终训练出了YiZhao-12B-Chat。YiZhao-12B-Chat具备自然语言理解、文本生成、舆情事件抽取、工具使用进行互动等多种功能。YiZhao-12B-Chat是一个专为金融领域设计的
120亿参数大型语言模型，支持32K上下文长度。
主要特点：

进行了多维度数据清洗与筛选,最终采用284GB金融语料与657GB通用语料进行训练，保证数据的量级和质量。

YiZhao-12B-Chat是基于GLM（General Language Model）架构的中英双语对话模型，具有120亿参数，专为问答和对话场景优化。

YiZhao-12B-Chat完全基于国产算力和国产深度学习框架MindSpore进行训练，算力和算法框架更自主可控。

这款模型在云端私有化部署后，可以为企业和个人提供高效、灵活的智能对话解决方案。

下载链接：

|  社区  | 下载地址                                                       |
|:----:|:-----------------------------------------------------------|
| 魔搭社区 | https://www.modelscope.cn/models/CMB_AILab/YiZhao-12B-Chat |
| 魔乐社区 | https://modelers.cn/models/Cmb_AIlab/YiZhao-12B-Chat       |
|  码云  | https://gitee.com/mindspore/mindformers                    |

## 效果评测

效果评测包括通用能力评测和金融领域评测。模型在保持通用能力的基础上进一步提升金融领域能力。

### 1. 通用评测

在通用领域评测中，我们选择当下主流的几类客观评测基准，见下表：

| 能力   | 任务                                                               | 描述                                                                               |
|------|------------------------------------------------------------------|----------------------------------------------------------------------------------|
| 逻辑推理 | [ARC Challenge](https://huggingface.co/datasets/malhajar/arc-tr) | ARC问题需要多种类型的知识与推理，包括定义、基本事实和属性、结构、过程与因果、目的论、代数、实验、空间/运动学、类比等，ARC问题集包含7787个自然科学问题 |
| 中文知识 | [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu)         | 67个学科中文考试题                                                                       |
| 数学能力 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)            | 8.5k个样例，数学推理能力                                                                   |
| 通用知识 | [MMLU](https://huggingface.co/datasets/cais/mmlu)                | MMLU 是一个涵盖STEM、人文学科、社会科学等57个学科领域(例如,数学、法律、伦理等)的评测基准,旨在考察大语言模型将知识运用于问题解决的能力       |
| 指令遵从 | [IFEval](https://huggingface.co/datasets/HuggingFaceH4/ifeval)   | 确定了25种可验证的指令类型，500个包含一个或多个可验证指令的提示（prompts）                                      |

#### 测试结果如下：

|     模型      |     逻辑推理      |  中文知识  |  数学能力  |  通用知识  |  指令遵从  |
|:-----------:|:-------------:|:------:|:------:|:------:|:------:|
|             | arc_challenge | cmmlu  | gsm8k  |  mmlu  | ifeval |
| 一招-12B-Chat |    0.9331     | 0.7158 | 0.8993 | 0.7192 | 0.5432 |

#### 小结：

YiZhao-12B在通用评测集方面均有出色表现。

YiZhao-12B的预训练方案并没有牺牲过多模型通用能力，而且增量训练数据中的中文金融数据，也一定程度地增强了模型的逻辑推理、中文、数学等能力。

### 2. 金融评测

金融评测主要包括以下三个测试：

| 任务                                                                                            | 描述                                                                                                                                                                                                                                     |
|:----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CCKS-IJCKG：DFMEB](https://sigkg.cn/ccks-ijckg2024/evaluation/)                               | 为了推动LLM在数字金融领域的发展，并解决实际金融业务问题。招商银行联合中科院自动化所、科大讯飞股份有限公司，结合实际生产场景，推出数字金融领域评测基准（Digital Finance Model Evaluation Benchmark，DFMEB）。该评测基准包含六大场景（知识问答、文本理解、内容生成、逻辑推理、安全合规、AI智能体），涵盖69种金融任务，有利于帮助开源社区和业界快速评测公开或者自研LLM。                        |
| [CFBenchmark-OpenFinData](https://github.com/TongjiFinLab/CFBenchmark/blob/main/README-CN.md) | “书生•济世”中文金融评测基准（CFBenchmark）基础版本由CFBenchmark-Basic（全部为主观题，不参与测评）和CFBenchmark-OpenFinData两部分数据组成。OpenFinData是由东方财富与上海人工智能实验室联合发布的开源金融评测数据集。该数据集代表了最真实的产业场景需求，是目前场景最全、专业性最深的金融评测数据集。它基于东方财富实际金融业务的多样化丰富场景，旨在为金融科技领域的研究者和开发者提供一个高质量的数据资源。 |
| [FinancelQ](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)                     | FinanceIQ是一个专业的金融领域评测集，其涵盖了10个金融大类及36个金融小类，总计7173个单项选择题，某种程度上可客观反应模型的金融能力。                                                                                                                                                             |

#### 测试结果如下：

|     模型      | CCKS-IJCKG：DFMEB | CFBenchmark-OpenFinData | FinancelQ |
|:-----------:|:----------------:|:-----------------------:|:---------:|
| 一招-12B-Chat |      0.8218      |         0.8798          |  0.6867   |

#### 小结：

YiZhao-12B-Chat在金融测评方面表现优异。
YiZhao-12B-Chat有着较强的专业知识能力，在金融分析、金融考核、金融安全合规、风险检查等多个专业领域维度有着极好的表现。

## 声明与协议

### 声明

我们在此声明，不要使用一招（YiZhao）模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将一招（YiZhao）用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用一招（YiZhao）开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

### 协议

社区使用一招（YiZhao）模型需要遵循[《“一招（YiZhao）”许可协议》](https://modelscope.cn/models/CMB_AILab/YiZhao-12B-Chat/file/view/master?fileName=LICENSE.txt&status=1)。

如果使用方为非自然人实体，您需要通过以下联系邮箱ailab@cmbchina.com，提交《“一招（YiZhao）”许可协议》要求的申请材料。审核通过后，将特此授予您一个非排他性、全球性、不可转让、不可再许可、可撤销和免版税的有限许可，但仅限于被许可方内部的使用、复制、分发以及修改，并且应当遵守《“一招（YiZhao）”许可协议》。

## 模型文件

`yizhao_12b` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

```text
    yizhao/
        ├── yizhao.py                        # yizhao模型文件
        ├── yizhao_config.py                 # yizhao配置文件
        ├── yizhao_dpo_dataset.py            # DPO训练数据集加载文件
        ├── yizhao_loss.py                   # 训练loss定义文件
        ├── yizhao_modules.py                # yizhao模型模块实现细节
        ├── yizhao_scheduler.py              # 学习率调度文件
        ├── yizhao_tokenizer.py              # tokenizer
        └── yizhao_transformer.py            # transformer实现

```

2. 模型配置：

```text
    yizhao/yizhao_12b
        ├── pretrain_yizhao_12b_8k.yaml                         # 预训练启动配置  
        ├── finetune_yizhao_12b_8k.yaml                         # 全参微调启动配置
        ├── reinforce_learning_yizhao_12b_4k_dpo.yaml           # DPO微调启动配置
        ├── yizhao_dpo_dataset.yaml                             # DPO数据集生成配置
        └── predict_yizhao_12b.yaml                             # YiZhao推理配置
```

3. 环境准备和任务启动脚本：

```text
    yizhao/
        ├── convert_reversed.py              # ckpt权重转pth权重
        ├── convert_weight.py                # pth权重转ckpt权重
        ├── alpaca_convert.py                # alpaca数据集格式转换脚本
        ├── alpaca_data_process.py           # alpaca数据集预处理
        ├── wiki_data_process.py             # wikitext数据预处理
        └── run_yizhao_chat.py               # 一招推理示例脚本
```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)
和[版本匹配关系](../../README_CN.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持yizhao-12b的预训练、全参微调、DPO微调。

### 数据及权重准备

#### 数据集下载

MindFormers提供`Wikitext-103`作为[预训练](#预训练)数据集，`alpaca`作为[微调](#全参微调)数据集， `DPO-En-Zh-20k`作为[DPO](#dpo微调)数据集。

| 数据集名称         |    适用模型    |   适用阶段   |                                            下载链接                                            |
|:--------------|:----------:|:--------:|:------------------------------------------------------------------------------------------:|
| Wikitext-103  | yizhao-12b | Pretrain | [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) |
| alpaca        | yizhao-12b | Finetune |      [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)       |
| DPO-En-Zh-20k | yizhao-12b |   DPO    |               [Link](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)                |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **Wikitext-103 数据预处理**

  使用`yizhao/wiki_data_process.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
  python wiki_data_process.py \
   --vocab_file /path/tokenizer.model \
   --ori_file_path /path/wiki.train.tokens \
   --output_file_path /path/wiki.mindrecord \
   --seq_length 8192 \
   --num_proc 32
  ```

  参数说明:
  - vocab_file:         tokenizer.model词表文件路径。
  - ori_file_path:      输入下载后wiki.train.tokens的文件路径。
  - output_file_path:   输出文件的保存路径。
  - seq_length:         输出数据的序列长度。
  - num_proc:           批处理进程数

- **alpaca 数据预处理**

  执行`yizhao/alpaca_convert.py`，将原始数据集转换为jsonl格式。

  ```shell
  python alpaca_convert.py \
   --data_path /path/alpaca_data.json \
   --output_path /path/alpaca_data.jsonl
  ```

  参数说明:
  - data_path:   输入下载的文件路径
  - output_path: 输出文件的保存路径

  执行`yizhao/alpaca_data_process.py`文件，进行数据预处理和Mindrecord数据生成。

  ```shell
  python alpaca_data_process.py \
   --vocab_file /path/tokenizer.model \
   --ori_data_file_path /path/alpaca_data.jsonl \
   --output_file /path/alpaca.mindrecord \
   --seq_length 8192 \
   --aggregated_multitask True \
   --num_proc 8
  ```

  参数说明:
  - vocab_file:              tokenizer.model词表文件路径
  - ori_data_file_path:      输入处理后alpaca_data.jsonl的文件路径
  - output_file_path:        输出文件的保存路径
  - seq_length:              输出数据的序列长度
  - aggregated_multitask:    是否将多个小于seq_length短样本合并
  - num_proc:                批处理进程数

#### 模型权重下载

用户可以从魔搭社区官方下载预训练权重，`tokenizer.modl`文件也在链接中下载。

词表下载链接：

| 模型名称       |                           权重                            |
|:-----------|:-------------------------------------------------------:|
| yizhao-12b | https://modelscope.cn/models/CMB_AILab/YiZhao-12B-Chat/ |

#### 模型权重转换

- **torch权重转mindspore权重**

  **注**: 请安装`convert_weight.py`依赖包。

  ```shell
  pip install torch transformers
  ```

  下载完成后，运行mindformers根目录的`convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

  ```shell
  python convert_weight.py \
   --model yizhao \
   --input_path <torch_ckpt_dir> \
   --output_path <mindspore_ckpt_path> \
   --config <mindformers_model_yaml> \
   --dtype bf16
  ```

  参数说明：
  - model:               模型名, 这里是yizhao
  - input_path:          预训练权重文件所在的目录
  - output_path:         转换后的输出文件存放路径
  - config:              mindformers模型文件配置yaml, 例如推理可以使用 yizhao/yizhao_12b/predict_yizhao_12b.yaml
  - dtype:               转换后权重文件格式

- **mindspore权重转torch权重**

  在生成mindspore权重之后如需使用torch运行，运行mindformers根目录的`convert_weight.py`转换脚本转换：

  ```shell
  python convert_weight.py \
   --model yizhao \
   --input_path <mindspore_ckpt_path> \
   --output_path <torch_ckpt_dir> \
   --config <mindformers_model_yaml> \
   --dtype bf16 \
   --reversed
  ```

  参数说明：
  - model:              模型名, 这里是yizhao
  - input_path:         待转换权重文件所在的目录
  - output_path:        转换后的输出文件存放路径
  - config:             mindformers模型文件配置yaml, 例如推理可以使用 yizhao/yizhao_12b/predict_yizhao_12b.yaml
  - dtype:              转换后权重文件格式
  - reversed:           mindspore转为pt权重的标志

- **[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)**

  从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

  通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

  以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 预训练

MindFormers提供`yizhao-12b`多机多卡的预训练示例，过程中使用`Wikitext-103`
数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

1. 启动yizhao-14b预训练，执行2机16卡任务。

   在多机上同时拉起任务，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`
   不同，具体可参考[使用指南](../../README_CN.md#三使用指南)

   在mindformers工作目录下，执行：

   ```shell
   # 节点0，节点ip示例为192.168.1.1，节点启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config yizhao/yizhao_12b/pretrain_yizhao_12b_8k.yaml \
    --register_path yizhao \
    --use_parallel True \
    --run_mode train \
    --load_checkpoint /path/yizhao.ckpt \
    --train_data /path/wiki.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 3000

   # 节点1，节点ip示例为192.168.1.2，节点启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config yizhao/yizhao_12b/pretrain_yizhao_12b_8k.yaml \
    --register_path yizhao \
    --use_parallel True \
    --run_mode train \
    --load_checkpoint /path/yizhao.ckpt \
    --train_data /path/wiki.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 3000
   ```

   参数说明:
   - config:           配置文件路径
   - run_mode:         运行模式, 预训练时设置为train
   - train_data:       训练数据集文件夹路径
   - load_checkpoint:  权重文件路径
   - register_path:    yizhao模型文件夹路径

## 全参微调

MindFormers提供`yizhao-12b`多机多卡的微调示例，过程中使用`alpaca`
数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 多机微调

以`yizhao-12b`2机16卡为例，启动多机微调任务。

1. 启动yizhao-14b全参微调，执行2机16卡任务。

   在多机上同时拉起任务，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`
   不同，具体可参考[使用指南](../../README_CN.md#三使用指南)

   在mindformers工作目录下，执行：

   ```shell
   # 节点0，节点ip为192.168.1.1，作为主节点，总共32卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config yizhao/yizhao_12b/finetune_yizhao_12b_8k.yaml \
    --register_path yizhao \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --train_data /path/alpaca.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300

   # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config yizhao/yizhao_12b/finetune_yizhao_12b_8k.yaml \
    --register_path yizhao \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --train_data /path/alpaca.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 300
   ```

   参数说明:
   - config:            配置文件路径
   - load_checkpoint:   权重文件路径
   - auto_trans_ckpt:   自动权重转换开关
   - run_mode:          运行模式, 微调时设置为finetune
   - train_data:        训练数据集路径
   - register_path:     yizhao模型文件夹路径

## DPO微调

MindFormers提供`yizhao-12b`单机多卡的DPO微调示例，过程中使用`DPO-En-Zh-20k`
数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机微调

以`yizhao-12b`单机八卡为例，启动DPO微调任务。

1. 启动yizhao-14b DPO微调，执行单机八卡任务。

   在mindformers工作目录下，执行：

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config yizhao/yizhao_12b/dpo_yizhao_12b_4k.yaml \
    --register_path yizhao \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --train_data /path/DPO.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300
   ```

   参数说明:
   - config:            配置文件路径
   - load_checkpoint:   权重文件路径
   - auto_trans_ckpt:   自动权重转换开关
   - run_mode:          运行模式, 微调时设置为finetune
   - train_data:        训练数据集路径
   - register_path:     yizhao模型文件夹路径

## 推理

提供了推理示例脚本 run_yizhao_chat.py，使用如下命令进行推理：

```shell
python run_yizhao_chat.py \
--config_path yizhao_12b/predict_yizhao_12b.yaml \
--load_checkpoint /path/to/model.ckpt \
--vocab_file /path/to/tokenizer.model
```
