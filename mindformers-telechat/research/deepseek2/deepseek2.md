# DeepSeekV2

## 模型描述

DeepSeek-V2采用了创新的MLA（Multi-head Latent Attention）注意力机制和DeepSeekMoE前馈网络，不仅大幅降低了计算量和显存占用，还显著提升了模型的推理效率。相比传统的Dense或Sparse结构，DeepSeek-V2在保持高性能的同时，实现了更低的资源消耗。

在训练过程中，DeepSeek-V2使用了高质量、多样化的8.1万亿token预训练语料库，并针对不同任务（如数学、编程、对话等）进行了监督微调和强化学习。这些优化措施确保了模型在训练成本上大幅优于同类模型，同时提升了模型的泛化能力和实际应用效果。

## 模型性能

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                           |      Task       |  Datasets   | SeqLength |  Phase   |  Performance  |
|:-------------------------------------------------|:---------------:|:-----------:|:---------:|:--------:|:-------------:|
| [deepseek2-236b](./finetune_deepseek2_236B.yaml) | text_generation | code_alpaca |   4096    | Finetune | 36 tokens/s/p |

## 模型文件

`deepseekv2-236b` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型配置：

    ```text
    deepseek2
        ├── convert_weight.py                   # pt2ms权重转换脚本
        ├── deepseek2.py                        # 模型主体
        ├── deepseek2_config.py                 # 模型配置类
        ├── deepseek2_layer.py                  # 模型个别模型层
        ├── deepseek2_transformer.py            # 模型主要模块
        ├── run_train_deepseek2_236B.yaml       # 全参微调启动配置
        └── run_predict_deepseek2_236B.yaml     # 在线推理启动配置
    ```

## 环境及数据准备

### 安装环境

### 环境参数设置

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 数据及权重准备

#### 数据集下载

[https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json]()

### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，tokenizer.json文件也在链接中下载。

| 模型名称                        |                             权重                             | 用图 |
| :------------------------------ | :----------------------------------------------------------: | ---- |
| DeepSeek-V2-Chat | [Link](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) | 多轮对话 |
| DeepSeek-V2     |       [Link](https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main)        | Base |

#### 模型权重转换

##### torch权重转mindspore权重

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```python
cd research/deepseek2
python convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
```

- 参数说明
  torch_ckpt_path: 预训练权重文件所在的目录
  mindspore_ckpt_path: 转换后的输出文件存放路径（`.ckpt`文件）

##### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)*

从hugging face或官方github仓库转换而来的权重通常是完整权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 全参微调

全参微调至少需要8机64卡，依次在每一台机器的项目根目录执行如下命令，其中127.0.0.1应当修改为第一个节点的主机ip：

```shell
//在所有节点设置环境变量
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800

//第0个节点
bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py \
--config run_train_deepseek2_236B.yaml \
--use_parallel True \
--load_checkpoint  ./ckpt_trans \
--run_mode train \
--train_data  ./dataset/train_data" \
64 8 127.0.0.1 9543 0 output/msrun_log False 3000

//第1个节点
bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py \
--config run_train_deepseek2_236B.yaml \
--use_parallel True \
--load_checkpoint  ./ckpt_trans \
--run_mode train \
--train_data  ./dataset/train_data" \
64 8 127.0.0.1 9543 1 output/msrun_log False 3000

...

//第6个节点
bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py \
--config run_train_deepseek2_236B.yaml \
--use_parallel True \
--load_checkpoint  ./ckpt_trans \
--run_mode train \
--train_data  ./dataset/train_data" \
64 8 127.0.0.1 9543 6 output/msrun_log False 3000

//第7个节点
bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py \
--config run_train_deepseek2_236B.yaml \
--use_parallel True \
--load_checkpoint  ./ckpt_trans \
--run_mode train \
--train_data  ./dataset/train_data" \
64 8 127.0.0.1 9543 7 output/msrun_log False 3000
```

参数说明：

- config: 固定路径，配置文件所在路径
- usr_parallel：固定值，True
- load_checkpoint：加载切分后权重的路径，具体到文件夹
- run_mode：固定值，train
- train_data：数据集所在位置，具体到文件夹

> 注：此模型暂不支持配置`context_parallel`，因此暂不支持长序列。

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合算大算子降低推理时延，有效提升网络吞吐量。

### 基于高阶接口推理

#### 参数配置

deepseekv2-236b模型推理至少需要两机16卡，依次在每一台机器的项目根目录执行如下命令，其中127.0.0.1应当修改为第一个节点的主机ip：

> 核查配置文件 `run_predict_deepseek2_236B.yaml`。
> `checkpoint_name_or_path`，`tokenizer_file`参数为待使用的真实配置地址。
> 核查无误进行后续操作。

#### 多卡推理

  ```shell
  cd research/deepseek2

  //第0个节点
  bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py --config run_predict_deepseek2_236B.yaml --run_mode=predict --predict_data 'An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is' --predict_length 128 --use_parallel True" 16 8 127.0.0.1 8421 0 output/msrun_log False 300

  //第1个节点
  bash ../../scripts/msrun_launcher.sh "../../run_mindformer.py --config run_predict_deepseek2_236B.yaml --run_mode=predict --predict_data 'An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is' --predict_length 128 --use_parallel True" 16 8 127.0.0.1 8421 0 output/msrun_log False 300
  ```
