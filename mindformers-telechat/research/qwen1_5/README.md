# 通义千问

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。Qwen1.5是Qwen2的beta版本, 基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

```text
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                      |      Task       |   Datasets   | SeqLength |  Phase   |   Performance   |
|:------------------------------------------------------------|:---------------:|:------------:|:---------:|:--------:|:---------------:|
| [qwen1.5-7b](qwen1_5_7b/finetune_qwen1_5_7b.yaml)           | text_generation |    alpaca    |   4096    | Finetune | 2684 tokens/s/p |
| [qwen1.5-7b](qwen1_5_7b/pretrain_qwen1_5_7b.yaml)           | text_generation | Wikitext-103 |   32768   | Pretrain | 1417 tokens/s/p |
| [qwen1.5-14b](qwen1_5_14b/finetune_qwen1_5_14b.yaml)        | text_generation |    alpaca    |   4096    | Finetune | 1452 tokens/s/p |
| [qwen1.5-0.5b](qwen1_5_0_5b/predict_qwen1_5_0_5b_chat.yaml) | text_generation |      -       |   8192    | Predict  |  1491 tokens/s  |
| [qwen1.5-1.8b](qwen1_5_1_8b/predict_qwen1_5_1_8b_chat.yaml) | text_generation |      -       |   4096    | Predict  |  1179 tokens/s  |
| [qwen1.5-4b](qwen1_5_4b/predict_qwen1_5_4b_chat.yaml)       | text_generation |      -       |   4096    | Predict  |  625 tokens/s   |
| [qwen1.5-7b](qwen1_5_7b/predict_qwen1_5_7b_chat.yaml)       | text_generation |      -       |   8192    | Predict  |  164 tokens/s   |
| [qwen1.5-14b](qwen1_5_14b/predict_qwen1_5_14b_chat.yaml)    | text_generation |      -       |   8192    | Predict  |  104 tokens/s   |
| [qwen1.5-32b](qwen1_5_32b/predict_qwen1_5_32b_chat.yaml)    | text_generation |      -       |   4096    | Predict  |  245 tokens/s   |
| [qwen1.5-72b](qwen1_5_72b/predict_qwen1_5_72b_chat.yaml)    | text_generation |      -       |   8192    | Predict  |   74 tokens/s   |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                               |      Task       |   Datasets   | SeqLength |  Phase   |   Performance    |
|:-----------------------------------------------------|:---------------:|:------------:|:---------:|:--------:|:----------------:|
| [qwen1.5-0.5b](qwen1_5_0_5b/finetune_qwen1_5_0_5b.yaml)         | text_generation |    alpaca    |   8192    | Finetune | 21171 tokens/s/p |
| [qwen1.5-1.8b](qwen1_5_1_8b/finetune_qwen1_5_1_8b.yaml)         | text_generation |    alpaca    |   8192    | Finetune | 11241 tokens/s/p |
| [qwen1.5-4b](qwen1_5_4b/finetune_qwen1_5_4b.yaml)             | text_generation |    alpaca    |   8192    | Finetune | 4844 tokens/s/p  |
| [qwen1.5-32b](qwen1_5_32b/finetune_qwen1_5_32b.yaml) | text_generation |    alpaca    |   8192    | Finetune |  671 tokens/s/p  |
| [qwen1.5-14b](qwen1_5_14b/pretrain_qwen1_5_14b.yaml) | text_generation | Wikitext-103 |   32768   | Pretrain |  787 tokens/s/p  |
| [qwen1.5-72b](qwen1_5_72b/pretrain_qwen1_5_72b.yaml) | text_generation | Wikitext-103 |   32768   | Pretrain |  183 tokens/s/p  |

## 模型文件

`Qwen1.5` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型实现：

   ```text
   research/qwen1_5
     └── qwen1_5_tokenizer.py          # 模型tokenizer
   ```

2. 模型配置：

   ```text
   research/qwen1_5
     ├── qwen1_5_0_5b                                        # qwen1.5 0.5B 配置文件
     │   ├── finetune_qwen1_5_0_5b.yaml                      # 0.5B 全参微调启动配置
     │   └── predict_qwen1_5_0_5b_chat.yaml                  # 0.5B 在线推理启动配置
     ├── qwen1_5_1_8b                                        # qwen1.5 1.8B 配置文件
     │   ├── finetune_qwen1_5_1_8b.yaml                      # 1.8B 全参微调启动配置
     │   └── predict_qwen1_5_1_8b_chat.yaml                  # 1.8B 在线推理启动配置
     ├── qwen1_5_4b                                          # qwen1.5 4B 配置文件
     │   ├── finetune_qwen1_5_4b.yaml                        # 4B 全参微调启动配置
     │   └── predict_qwen1_5_4b_chat.yaml                    # 4B 在线推理启动配置
     ├── qwen1_5_7b                                          # qwen1.5 7B 配置文件
     │   ├── finetune_qwen1_5_7b.yaml                        # 7B 全参微调启动配置
     │   ├── pretrain_qwen1_5_7b.yaml                        # 7B 训练启动配置
     │   ├── predict_qwen1_5_7b.yaml                         # 7B 在线推理启动配置
     │   └── predict_qwen1_5_7b_chat.yaml                    # 7B 多轮对话推理启动配置
     ├── qwen1_5_14b                                         # qwen1.5 14B 配置文件
     │   ├── finetune_qwen1_5_14b.yaml                       # 14B 全参微调启动配置
     │   ├── pretrain_qwen1_5_14b.yaml                       # 14B 训练启动配置
     │   ├── predict_qwen1_5_14b.yaml                        # 14B 在线推理启动配置
     │   └── predict_qwen1_5_14b_chat.yaml                   # 14B 多轮对话推理启动配置
     ├── qwen1_5_32b                                         # qwen1.5 32B 配置文件
     │   ├── finetune_qwen1_5_32b.yaml                       # 32B 全参微调启动配置
     │   └── predict_qwen1_5_32b_chat.yaml                   # 32B 在线推理启动配置
     ├── qwen1_5_72b                                         # qwen1.5 72B 配置文件
     │   ├── finetune_qwen1_5_72b.yaml                       # 72B 全参微调启动配置
     │   ├── pretrain_qwen1_5_72b.yaml                       # 72B 训练启动配置
     │   ├── predict_qwen1_5_72b.yaml                        # 72B 在线推理启动配置
     └── └── predict_qwen1_5_72b_chat.yaml                   # 72B 多轮对话推理启动配置
   ```

3. 模型相关脚本：

   ```text
   research/qwen1_5
     ├── alpaca_converter.py           # alpaca数据集格式转换脚本
     ├── qwen1_5_preprocess.py         # 数据集预处理脚本
     ├── convert_weight.py             # hf->ms权重转换脚本
     ├── convert_reversed.py           # ms->hf权重转换脚本
     ├── qwen1_5_chat.py               # Qwen1_5 Chat功能函数
     └── run_qwen1_5_chat.py           # Qwen1_5 Chat功能启动脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供`Wikitext-103`作为[预训练](#预训练)数据集，`alpaca`作为[微调](#微调)数据集。

| 数据集名称        |                      适用模型                      |   适用阶段   |                                            下载链接                                            |
|:-------------|:----------------------------------------------:|:--------:|:------------------------------------------------------------------------------------------:|
| Wikitext-103 | qwen1_5-7b <br/> qwen1_5-14b <br/> qwen1_5-72b | Pretrain | [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) |
| alpaca       | qwen1_5-7b <br/> qwen1_5-14b <br/> qwen1_5-72b | Finetune |      [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)       |

数据预处理中所用的`vocab.json`和`merges.txt`可以参考[模型权重下载](#模型权重下载)进行下载。

- **Wikitext-103 数据预处理**

  使用`research/qwen1_5/qwen1_5_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
  python qwen1_5_preprocess.py \
   --dataset_type 'wiki' \
   --input_glob /path/wiki.train.tokens \
   --vocab_file /path/vocab.json \
   --merges_file /path/merges.txt \
   --seq_length 32768 \
   --output_file /path/wiki.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.train.tokens的文件路径
  vocab_file:   vocab.json文件路径
  merges_file:  merges.txt文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

- **alpaca 数据预处理**

  执行`research/qwen1_5/alpaca_converter.py`，将原始数据集转换为指定格式。

  ```shell
  python alpaca_converter.py \
   --data_path path/alpaca_data.json \
   --output_path /path/alpaca-data-messages.json

  # 参数说明
  data_path:   输入下载的文件路径
  output_path: 输出文件的保存路径
  ```

  执行`research/qwen1_5/qwen1_5_preprocess.py`文件，进行数据预处理和Mindrecord数据生成。

  ```shell
  python qwen1_5_preprocess.py \
   --dataset_type 'qa' \
   --input_glob /path/alpaca-data-messages.json \
   --vocab_file /path/vocab.json \
   --merges_file /path/merges.txt \
   --seq_length 4096 \
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

词表下载链接：[vocab.json](https://huggingface.co/Qwen/Qwen1.5-7B-Chat/blob/main/vocab.json)和[merges.txt](https://huggingface.co/Qwen/Qwen1.5-7B-Chat/blob/main/merges.txt)

| 模型名称        |                     Base权重（建议训练和微调使用）                     |                         Chat权重（建议推理使用）                         |
|:------------|:---------------------------------------------------------:|:--------------------------------------------------------------:|
| qwen1_5-7b  | [Link](https://huggingface.co/Qwen/Qwen1.5-7B/tree/main)  | [Link](https://huggingface.co/Qwen/Qwen1.5-7B-Chat/tree/main)  |
| qwen1_5-14b | [Link](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main) | [Link](https://huggingface.co/Qwen/Qwen1.5-14B-Chat/tree/main) |
| qwen1_5-72b | [Link](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main) | [Link](https://huggingface.co/Qwen/Qwen1.5-72B-Chat/tree/main) |

#### 模型权重转换

- **torch权重转mindspore权重**

  **注**: 请安装`convert_weight.py`依赖包。

  ```shell
  pip install torch transformers==4.37.2
  ```

然后运行 [Mindformers 的权重转换工具](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html), 将huggingface的权重转换为 Mindspore 的ckpt格式。

> 注意: 权重转换完成之后，注意重新根据本项目[requirements.txt](../../requirements.txt )恢复`tokenizers`包的版本:
> `pip install -r requirements.txt`

## 预训练

MindFormers提供`qwen1_5-7b`单机多卡以及`qwen1_5-14b`与`qwen1_5-72b`多机多卡的预训练示例，过程中使用`Wikitext-103`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机训练

以`qwen1_5-7b`单机8卡预训练任务为例，执行分布式启动脚本。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen1_5 \
 --config research/qwen1_5/qwen1_5_7b/pretrain_qwen1_5_7b.yaml \
 --load_checkpoint /path/qwen1.5_7b.ckpt \
 --train_dataset_dir /path/wiki.mindrecord \
 --run_mode train" 8
```

### 多机训练

1. 启动qwen1_5-14b预训练，执行2机16卡任务。

   在多机上同时拉起任务，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)

   在mindformers工作目录下，执行：

   ```shell
   # 节点0，节点ip为192.168.1.1，节点启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_14b/pretrain_qwen1_5_14b.yaml \
    --use_parallel True \
    --run_mode train \
    --train_data /path/wiki.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 3000

   # 节点1，节点ip为192.168.1.2，节点启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_14b/pretrain_qwen1_5_14b.yaml \
    --use_parallel True \
    --run_mode train \
    --train_data /path/wiki.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 3000

   # 参数说明
   config:      配置文件路径
   run_mode:    运行模式, 预训练时设置为train
   train_data:  训练数据集文件夹路径
   merges_file: 词表文件merges.txt路径
   vocab_file:  词表文件vocab.json路径
   ```

2. 启动qwen1_5-72b预训练，执行8机64卡任务。

  在多机上同时拉起任务，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)

  在mindformers工作目录下，执行：

   ```shell
   # 节点0，节点ip为192.168.1.1，作为主节点，总共64卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_72b/pretrain_qwen1_5_72b.yaml \
    --use_parallel True \
    --run_mode train \
    --train_data /path/wiki.mindrecord" \
   64 8 192.168.1.1 8118 0 output/msrun_log False 1200

   # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_72b/pretrain_qwen1_5_72b.yaml \
    --use_parallel True \
    --run_mode train \
    --train_data /path/wiki.mindrecord" \
   64 8 192.168.1.1 8118 1 output/msrun_log False 1200

   # ...
   # 省略中间节点2-6的执行命令不同节点之间仅参数NODE_RANK不同

   # 节点7，节点ip为192.168.1.8，节点0与节点7启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_72b/pretrain_qwen1_5_72b.yaml \
    --use_parallel True \
    --run_mode train \
    --train_data /path/wiki.mindrecord" \
   64 8 192.168.1.1 8118 7 output/msrun_log False 1200

   # 参数说明
   config:      配置文件路径
   run_mode:    运行模式, 预训练时设置为train
   train_data:  训练数据集文件夹路径
   merges_file: 词表文件merges.txt路径
   vocab_file:  词表文件vocab.json路径
   ```

## 全参微调

MindFormers提供`qwen1_5-7b`与`qwen1_5-14b`单机多卡以及`qwen1_5-72b`多机多卡的微调示例，过程中使用`alpaca`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

设置如下环境变量：

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
# 如出现OOM需要配置:
export ENABLE_CELL_RESUSE=1          # 打开内存复用
export MS_GE_ATOMIC_CLEAN_POLICY=1   # 打开内存优化
   ```

### 单机训练

以`qwen1_5-7b`单机8卡微调为例，使用配置文件`research/qwen1_5/qwen1_5_7b/finetune_qwen1_5_7b.yaml`。

执行如下命令启动微调任务。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen1_5 \
 --config research/qwen1_5/qwen1_5_7b/finetune_qwen1_5_7b.yaml \
 --load_checkpoint /path/qwen1.5_7b.ckpt \
 --auto_trans_ckpt True \
 --train_dataset /path/alpaca.mindrecord \
 --run_mode finetune" 8
```

`qwen1_5-7b`单机8卡微调任务替换命令中的`--load_checkpoint /path/qwen1.5_14b.ckpt`以及配置文件`research/qwen1_5/qwen1_5_14b/finetune_qwen1_5_14b.yaml`即可。

### 多机训练

以`qwen1_5-72b`4机32卡为例，启动多机微调任务。

1. 修改`research/qwen1_5/qwen1_5_72b/finetune_qwen1_5_72b.yaml`

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 4
     micro_batch_num: 48
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

2. 执行分布式启动命令

   在多机上同时拉起任务，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)

   在mindformers工作目录下，执行：

   ```shell
   # 节点0，节点ip为192.168.1.1，作为主节点，总共32卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config finetune_qwen1_5_72b.yaml \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --auto_trans_ckpt True \
    --train_data /path/alpaca.mindrecord" \
   32 8 192.168.1.1 8118 0 output/msrun_log False 300

   # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config finetune_qwen1_5_72b.yaml \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --auto_trans_ckpt True \
    --train_data /path/alpaca.mindrecord" \
   32 8 192.168.1.1 8118 1 output/msrun_log False 300

   # 节点2，节点ip为192.168.1.3，节点0与节点2启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config finetune_qwen1_5_72b.yaml \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --auto_trans_ckpt True \
    --train_data /path/alpaca.mindrecord" \
   32 8 192.168.1.1 8118 2 output/msrun_log False 300

   # 节点3，节点ip为192.168.1.4，节点0与节点3启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config finetune_qwen1_5_72b.yaml \
    --load_checkpoint /path/model_dir \
    --use_parallel True \
    --run_mode finetune \
    --auto_trans_ckpt True \
    --train_data /path/alpaca.mindrecord" \
   32 8 192.168.1.1 8118 3 output/msrun_log False 300

   # 参数说明
   config:          配置文件路径
   load_checkpoint: 权重文件夹路径, 权重按照'model_dir/rank_0/xxx.ckpt'格式存放
   auto_trans_ckpt: 自动权重转换开关
   run_mode:        运行模式, 微调时设置为finetune
   train_data:      训练数据集路径
   ```

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

注意事项：

当前支持模型已提供推理相关配置文件，请根据实际使用模型更改配置文件。

### 基于高阶接口的推理

#### 单卡推理

以`qwen1_5_7b`单卡推理为例，即使用 `research/qwen1_5/qwen1_5_7b/predict_qwen1_5_7b.yaml` 配置文件。

修改配置文件 `research/qwen1_5/qwen1_5_7b/predict_qwen1_5_7b.yaml` ：

```yaml
model:
  model_config:
    qkv_concat: False
processor:
  tokenizer:
    vocab_file: "/path/vocab.json"
    merges_file: "/path/merges.txt"
```

执行如下命令进行推理。

```shell
# 推理命令中参数会覆盖yaml文件中的相同参数
python run_mindformer.py \
 --register_path research/qwen1_5 \
 --config research/qwen1_5/qwen1_5_7b/predict_qwen1_5_7b.yaml \
 --load_checkpoint /path/model_dir \
 --run_mode predict \
 --use_parallel False \
 --auto_trans_ckpt False \
 --predict_data '帮助我制定一份去上海的旅游攻略'
# 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
```

#### 多卡推理

以`qwen1_5_72b`4卡推理为例，即使用 `research/qwen1_5/qwen1_5_72b/predict_qwen1_5_72b.yaml` 配置文件。

修改配置文件 `research/qwen1_5/qwen1_5_72b/predict_qwen1_5_72b.yaml` ：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   model:
     model_config:
     qkv_concat: False
   processor:
     tokenizer:
     vocab_file: "/path/vocab.json"
     merges_file: "/path/merges.txt"
   ```

   *注*：可配置`model_config:param_init_type`为`float32`提高推理精度，但同时会影响在线推理性能。

启动多卡推理：

   ```shell
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen1_5 \
    --config research/qwen1_5/qwen1_5_72b/predict_qwen1_5_72b.yaml \
    --load_checkpoint /path/model_dir \
    --run_mode predict \
    --use_parallel True \
    --auto_trans_ckpt True \
    --predict_data 帮助我制定一份去上海的旅游攻略" 4

   # 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
   ```

### 多轮对话推理

`run_qwen1_5_chat.py` 基于`model.generate()`实现，支持交互式多轮对话，支持加载lora权重、权重转换、多卡推理，暂不支持 batch 推理。

#### 单卡推理

以`qwen1_5_7b`单卡推理为例，即使用 `research/qwen1_5/qwen1_5_7b/predict_qwen1_5_7b_chat.yaml`

修改配置文件 `research/qwen1_5/qwen1_5_7b/predict_qwen1_5_7b_chat.yaml` ：

```yaml
model:
  model_config:
    qkv_concat: False
processor:
  tokenizer:
    vocab_file: "/path/vocab.json"
    merges_file: "/path/merges.txt"
```

执行如下命令进行多轮对话推理。

```shell
cd research/qwen1_5
python run_qwen1_5_chat.py \
 --config qwen1_5_7b/predict_qwen1_5_7b_chat.yaml \
 --load_checkpoint /path/to/qwen1_5_7b_chat.ckpt \
 --enable_history True \
 --use_parallel False \
 --auto_trans_ckpt False \
 --run_demo True \
 --device_id 0

# 参数说明
# --enable_history: 是否将历史对话带入后面的输入。在交互式模式下（且启动时指定了--enable_history=True），可以用 /clear 清除前面的对话历史，开始新一轮会话;
# --run_demo: 启动时是否自动运行预设的若干个问题（用于演示/试验目的）;
# --predict_data: 提交给模型进行推理的问题（run_qwen1_5_chat.py会将历史对话和问题按照chatml格式组装后提交给模型进行推理），可以给出多个问题。不给出此参数时，`run_qwen1_5_chat.py`按交互模式运行;
```

#### 多卡推理

注意: 多卡运行`run_qwen1_5_chat.py`时，不支持交互式对话，只能通过`--predict_data`传入预先给出的问题。

修改配置文件 `research/qwen1_5/qwen1_5_72b/predict_qwen1_5_72b_chat.yaml` ：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   model:
     model_config:
     qkv_concat: False
   processor:
     tokenizer:
     vocab_file: "/path/vocab.json"
     merges_file: "/path/merges.txt"
   ```

```shell
cd research/qwen1_5
bash ../../scripts/msrun_launcher.sh "run_qwen1_5_chat.py \
 --config qwen1_5_72b/predict_qwen1_5_72b_chat.yaml \
 --use_parallel True \
 --auto_trans_ckpt False \
 --load_checkpoint /path/to/预先切分好的4卡权重 \
 --predict_data 《三体》这本小说的精彩之处在什么地方 再推荐几部刘慈欣的作品吧 国内这些年还有哪些不错的科幻作家 \
 --enable_history True" 4
tail -f output/msrun_log/*.log  # press Ctrl-C to quit when done
```
