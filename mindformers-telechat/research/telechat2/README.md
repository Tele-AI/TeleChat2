# 星辰语义大模型 TeleChat2

## 模型描述

- 星辰语义大模型**TeleChat2**是由中国电信人工智能研究院研发训练的大语言模型，包含7B, 35B, 115B三种规模，该系列模型**完全基于国产算力**训练。
- 本次开源**TeleChat2-115B**模型采用10万亿 Tokens中英文高质量语料进行训练，同步开源对话模型**TeleChat2-115B**的多格式、多平台权重文件。
- **TeleChat2**在训练数据、训练方法等方面进行了改进，在通用问答和知识类、代码类、数学类榜单上相比**TeleChat1**均有大幅提升。
    - **TeleChat2**完全基于国产算力和国产深度学习框架进行训练，算力和算法框架更自主可控。优化MP、PP、SP实现方式提升模型性能，优化算子来提升训练速度。
    - 我们使用大量小模型实验来验证scaling law规律，在不同模型结构、不同数据配比和数据清洗方式中寻找最优设计。
    - 采用RingAttention及其他序列切分方式，实现长文训练性能提升；通过ntk-aware+attention-scaling的方式保证训练长度切换时的平稳过渡，以此来保证模型在不同长度数据下的训练效果。
- 在微调数据方面，我们进行了指令复杂性提升与多样性扩充，通过数据合成和人工标注生成高质量数据，并使用拒绝采样生成多样的推理路径；通过研究一套基于base模型反向选择偏好对齐数据方案，基于适配数据最大限度提升模型效果。
    - 通用能力较TeleChat1系列模型提升超过29%，在逻辑推理、总结摘要、长文写作和数学计算上均有大幅提升。

基于GPU，Torch版本的TeleChat2链接：

[TeleChat2](https://github.com/Tele-AI/TeleChat2)

[TeleChat Technical Report](https://arxiv.org/abs/2401.03804)

``` text
@article{wang2024telechat,
      title={TeleChat Technical Report},
      author={Zihan Wang and Xinzhang Liu and Shixuan Liu and Yitong Yao and Yuyao Huang and Zhongjiang He and Xuelong Li and Yongxiang Li and Zhonghao Che and Zhaoxi Zhang and Yan Wang and Xin Wang and Luwen Pu and Huihan Xu and Ruiyu Fang and Yu Zhao and Jie Zhang and Xiaomeng Huang and Zhilong Lu and Jiaxin Peng and Wenjun Zheng and Shiquan Wang and Bingkai Yang and Xuewei he and Zhuoru Jiang and Qiyi Xie and Yanhan Zhang and Zhongqiu Li and Lingling Shi and Weiwei Fu and Yin Zhang and Zilu Huang and Sishi Xiong and Yuxiang Zhang and Chao Wang and Shuangyong Song},
      journal={arXiv preprint arXiv:2401.03804},
      year={2024}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

TeleChat2-7b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|-----------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [TeleChat2_7b](./run_telechat_115b_finetune.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 2950 tokens/s/p |
| [TeleChat2_7b](./run_telechat_115b_predict.yaml)  | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 54.1 tokens/s   |

TeleChat2-35b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|-----------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [TeleChat2_35b](./run_telechat_115b_finetune.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 516 tokens/s/p |
| [TeleChat2_35b](./run_telechat_115b_predict.yaml)  | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 27.7 tokens/s   |

TeleChat2-115b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|-----------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [TeleChat2_115b](./run_telechat_115b_finetune.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 158 tokens/s/p |
| [TeleChat2_115b](./run_telechat_115b_predict.yaml)  | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 26.5 tokens/s   |

## 模型文件

`TeleChat2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/research/telechat2`

   ```bash
   telechat
       ├── convert_weight.py                     # torch->ms权重转换脚本
       ├── convert_reversed.py                   # ms->torch权重转换脚本
       ├── telechat_preprocess.py                # telechat模型的mindrecord数据处理脚本
       ├── telechat.py                           # 模型实现
       ├── telechat_config.py                    # 模型配置项
       ├── telechat_layer.py                     # telechat网络层定义
       ├── telechat_interleave.py                # telechat细粒度多副本
       ├── telechat_predict_utils.py             # telechat推理模块
       ├── telechat_tokenizer.py                 # telechat tokenizer
       └── telechat_transformer.py               # transformer层实现
   ```

2. 模型配置：`mindformers/research/telechat2`

   ```bash
   telechat
       ├── finetune_telechat_7b.yaml             # 7b全量微调启动配置
       ├── predict_telechat_7b.yaml              # 7b推理启动配置
       ├── finetune_telechat_35b.yaml            # 35b全量微调启动配置
       ├── predict_telechat_35b.yaml             # 35b推理启动配置
       ├── finetune_telechat_115b.yaml           # 115b全量微调启动配置
       └── predict_telechat_115b.yaml            # 115b推理启动配置
   ```

3. 任务启动脚本：`mindformers/research/telechat2`

   ```text
   telechat
       ├── run_telechat_predict.py              # 推理脚本
       └── run_telechat.py                      # telechat高阶接口使用脚本
   ```

## 环境及数据准备

### 安装环境

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：Atlas 800T A2芯片支持telechat_115B单机多卡推理，至少使用8卡，全参微调至少需要8机64卡。

### 数据及权重准备

#### 数据集下载

TeleChat2全系列模型中7B，35B，115B所使用的微调数据集是由中电信人工智能科技有限公司所提供。

step 1. 获取数据集

[数据集]

数据集的格式：

```text
# input_dataset examples:
    {"input": "<_user>电信主卡和副卡的区别在哪里？<_bot>主卡和副卡的主要区别在于，主卡只能使用一张手机号码。<_end><_user>好的谢谢<_bot>很高兴为您服务<_end><_pad><_pad><_pad>"}
```

step 2. 处理数据成mindrecord格式

```bash
# 使用mindformers/research/telechat/telechat_preprocess.py进行数据预处理和Mindrecord数据生成
python telechat_preprocess.py \
--input_dataset_file /{path}/ \
--vocab_file_path /{path}/tokenizer.model \
--max_length 8192 \
--output_path /{path}/
```

```text
# 参数说明
input_dataset_file: 预训练的数据集
vocab_file_path: 词模型文件路径(如使用上述链接下载，指定到对应路径下即可)
max_length: 数据集长度
output_path: 生成数据集的路径
```

  > 注：`bos`, `eos`, `pad`等特殊`ids`要和`yaml`配置文件中`model_config`部分保持一致，默认`bos_token_id=1`, `eos_token_id=2`, `pad_token_id=3`。
如果有所修改，配置文件中对应设置也需要修改，通常预训练数据不包含`pad_token`，因此建议设置`pad_token_id=-1`。

#### 模型权重下载与转换

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1.torch模型权重及词模型下载链接：

- [TeleChat2-7b](https://modelscope.cn/models/TeleAI/TeleChat2-7B)
- [TeleChat2-35b](https://modelscope.cn/models/TeleAI/TeleChat2-35B)
- [TeleChat2-115b](https://modelscope.cn/models/TeleAI/TeleChat2-115B)

下载完成后，运行如下转换脚本，将全量微调的权重转换为完整的ckpt权重。

```shell
python mindformers/research/telechat2/convert_weight.py \
--torch_path TORCH_CKPT_DIR \
--mindspore_path {path} \
```

```text
# 参数说明
torch_path: torch版本权重保存目录路径
mindspore_path: 权重保存文件名，可以指定自定义保存路径
```

2.获取MindFormers提供的已转换权重，可直接从下面的链接获取。

- [TeleChat2-7b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_7B/Telechat_7B.zip)
- [TeleChat2-35b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_35B/Telechat_35B.zip)
- [TeleChat2-115b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_115B/Telechat_115B.zip)

### [分布式训练/微调权重合并](../../docs/feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix telechat_115B
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 微调

MindFormers提供`TeleChat2-115B`的微调示例，过程中使用中电信人工智能科技有限公司提供的数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 多机训练

- step 1. 修改模型对应的配置文件。

在模型对应的配置文件`research/telechat2/finetune_telechat_115b.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练)，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

1. 增加脚本入参`--load_checkpoint /{path}/telechat_115b.ckpt`加载预训练权重
2. 设置启动脚本中的`--train_dataset_dir /{path}/dataset.mindrecord`加载微调数据集
3. 设置启动脚本中的`--run_mode finetune`

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。auto_parallel说明详见[自动并行](../../docs/feature_cards/Auto_Parallel.md)。

- step 2. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以telechat-115b模型8机64卡训练为例，默认配置机4096卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：finetune_telechat_115b.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 8
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step3. 设置环境变量，变量配置如下：

```bash
export ENABLE_CELL_REUSE=1  #编译加速
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=InferenceMatmulSplit,PagedAttention
```

- step 4. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。

```shell
cd mindformers/

# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "python research/telechat2/run_telechat.py \
 --config research/telechat2/finetune_telechat_115b.yaml \
 --train_dataset /{path}/dataset.mindrecord \
 --use_parallel True \
  16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "python research/telechat2/run_telechat.py \
 --config research/telechat2/finetune_telechat_115b.yaml \
 --train_dataset /{path}/dataset.mindrecord \
 --use_parallel True \
  16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

```text
# 参数说明
config: 配置文件路径
train_dataset: 训练数据集文件夹路径
use_parallel：开启并行训练
```

## 推理

推理时所需的模型词表可在[模型权重下载与转换](#模型权重下载与转换)章节中下载得到，对应文件为`tokenizer.model`。此外，推理还需要用户自定义`input.json`文件，格式如下：

```json
{"input": "生抽和老抽的区别？"}
```

### 参数配置

- 设置环境变量：

export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=InferenceMatmulSplit,PagedAttention

- 7b模型支持单机**单卡推理**

在`predict_telechat_7b.yaml`中填写`vocab_file`字段

```yaml
processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

- 35b模型支持单机**2卡推理**

在`predict_telechat_35b.yaml`中填写`vocab_file`字段

```yaml
processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

- 115b模型支持单机**8卡推理**

在`predict_telechat_115b.yaml`中填写`vocab_file`字段

```yaml
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

### 启动推理

- 7b模型单卡推理

运行`run_mindformer.py`启动推理

```shell
cd mindformers
python research/telechat2/run_telechat_predict.py \
--yaml_file ./research/telechat2/predict_telechat_7b.yaml \
--checkpoint_path path/to/ckpt_path \
--use_parallel False \
--input_file input.json
```

- 35b模型2卡推理

```shell
cd mindformers/
bash scripts/msrun_launcher.sh "python ./research/telechat2/run_telechat_predict.py \
--yaml_file ./research/telechat2/predict_telechat_35b.yaml \
--checkpoint_path path/to/ckpt_path \
--input_file input.json \
--auto_trans_ckpt True \
--use_parallel True" \
2
```

- 115b模型8卡推理

```shell
cd mindformers/
bash scripts/msrun_launcher.sh "python ./research/telechat2/run_telechat_predict.py \
--yaml_file ./research/telechat2/predict_telechat_115b.yaml \
--checkpoint_path path/to/ckpt_path \
--input_file input.json \
--auto_trans_ckpt True \
--use_parallel True" \
8
```

```text
# 参数说明
yaml_file: 模型的配置文件
checkpoint_path: 权重路径
input_file: 输入的问题的文件路径
auto_tans_ckpt: 权重自动转换开关
use_parallel: 并行模式开关
```

### 推理结果

115B 模型推理结果如下：

```text
生抽与老抽的区别？

生抽和老抽是两种不同的酱油，它们在风味、色泽和用途上都有所区别。

1.颜色：生抽的颜色比较淡，而老抽的颜色较深。生抽的颜色呈红褐色或棕红色，而老抽的颜色则呈棕黑色。

2.味道：生抽具有鲜美的咸味和微甜的味浅，而老抽浓郁，颜色较深。根据个人口味和烹饪需求选择不同的酱油类型可以获得更好的口感和菜肴效果。
```
