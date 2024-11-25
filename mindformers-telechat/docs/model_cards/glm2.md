# ChatGLM2

## 模型描述

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B引入了新特征：**更强大的性能**、**更长的上下文**、**更高效的推理**、**更开放的协议**。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 模型性能

- 以下模型性能均由Atlas 800硬件环境下测试得出。

GLM2_6b:

| Config                                                               | Task            | Datasets | Metric                                  | Phase               | Score                                  | Performance                                    |
|----------------------------------------------------------------------|-----------------|----------|-----------------------------------------|---------------------|----------------------------------------|------------------------------------------------|
| [glm2_6b](../../configs/glm2/run_glm2_6b_finetune_800T_A2_64G.yaml)  | text_generation | ADGEN    | -                                       | [finetune](#微调)     | -                                      | 815.2059134 tokens/s/p                         |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora_800T_A2_64G.yaml) | text_generation | ADGEN    | -                                       | [finetune](#lora微调) | -                                      | 3243.697479 tokens/s/p                         |
| [glm2_6b](../../configs/glm2/run_glm2_6b_finetune_eval.yaml)         | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)         | 30.7842<br>7.0734<br>24.7739<br>7.4661 | -                                              |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora_eval.yaml)        | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)         | 31.0563<br>7.1753<br>24.2296<br>7.2294 | -                                              |
| [glm2_6b](../../configs/glm2/predict_glm2_6b.yaml)                   | text_generation | -        | -                                       | [predict](#推理)      | -                                      | 32.08 tokens/s (use_past=True, seq_length=512) |

## 模型文件

`chatGLM2-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm2
        ├── __init__.py
        ├── glm2.py                  # 模型实现
        ├── glm2_config.py           # 模型配置项
        ├── glm2_modules.py          # 模组实现
        ├── glm2_tokenizer.py        # tokenizer
        └── glm2_transformer.py      # transformer层实现
    ```

2. 模型配置：

    ```text
    configs/glm2
      ├── run_glm2_6b.yaml
      ├── run_glm2_6b_finetune_2k_800T_A2_64G.yaml  # Atlas 800T A2 最佳性能全量微调启动配置
      ├── run_glm2_6b_finetune_2k_800_32G.yaml      # Atlas 800 最佳性能全量微调启动配置
      ├── run_glm2_6b_finetune_800T_A2_64G.yaml     # Atlas 800T A2 ADGEN全量微调启动配置
      ├── run_glm2_6b_finetune_800_32G.yaml         # Atlas 800 ADGEN全量微调启动配置
      ├── run_glm2_6b_finetune_eval.yaml            # 全量微调后评估配置
      ├── run_glm2_6b_lora_2k_800T_A2_64G.yaml      # Atlas 800T A2最佳性能 lora微调启动配置
      ├── run_glm2_6b_lora_2k_800_32G.yaml          # Atlas 800 最佳性能 lora微调启动配置
      ├── run_glm2_6b_lora_800T_A2_64G.yaml         # Atlas 800T A2 ADGEN lora微调启动配置
      ├── run_glm2_6b_lora_800_32G.yaml             # Atlas 800 ADGEN lora微调启动配置
      └── run_glm2_6b_lora_eval.yaml                # lora微调评估配置
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

模型使用[`ADGEN`](https://aclanthology.org/D19-1321.pdf)（广告生成）数据集作为微调数据集。

| 数据集名称 |    适用模型     |   适用阶段   |                                下载链接                                |
|:------|:-----------:|:--------:|:------------------------------------------------------------------:|
| ADGEN | ChatGLM2-6b | Finetune | [Link](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) |

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenizer.model)

| 模型名称            |                                                   MindSpore权重                                                   |                  HuggingFace权重                   |
|:----------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------:|
| ChatGLM2-6b     |                                                        /                                                        | [Link](https://huggingface.co/THUDM/chatglm2-6b) |

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model glm-n --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 微调

MindFormers提供`ChatGLM2-6B`的微调示例， 过程中使用`ADGEN`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

全参微调使用`configs/glm2/run_glm2_6b_finetune_800T_A2_64G.yaml`配置文件，配置文件中定义了微调所需的各配置项。

> 注：微调时模型的`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。 在配置文件中默认的`seq_length: 192`以及`max_source_length: 64`和`max_target_length: 127`适用于`ADGEN`数据集，
> 对于其他数据集，可以将数据集转换为`token_id`，使`seq_length`等于`token_id`的最大长度，`seq_length`太大影响训练性能，太小影响训练精度，需要做出权衡。

以`glm2_6b`单机8卡微调为例。

1. 修改配置文件`configs/glm2/run_glm2_6b_finetune_800T_A2_64G.yaml`

   ```yaml
   train_dataset: &train_dataset
     tokenizer:
       type: ChatGLM2Tokenizer
       vocab_file: "/path/to/tokenizer.model"
   ```

2. 执行训练命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/glm2/run_glm2_6b_finetune_800T_A2_64G.yaml \
    --load_checkpoint {path}/glm2_6b.ckpt \
    --train_dataset_dir {path}/AdvertiseGen/train.json \
    --use_parallel True \
    --run_mode finetune"
   ```

补充说明：

1. 训练的log日志路径：`mindformers/output/log`
2. checkpoint(含优化器参数)存储路径：`mindformers/output/checkpoint`
3. checkpoint(不含优化器参数)存储路径：`mindformers/output/checkpoint_network`
4. 若想合并ckpt用于后续评估，选择不含优化器参数的权重即可

### LoRA微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象。 因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象。

使用LoRA算法进行低参微调时，使用 `configs/glm2/run_glm2_6b_lora_800T_A2_64G.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项。

1. 修改配置文件`configs/glm2/run_glm2_6b_lora_800T_A2_64G.yaml`

   ```yaml
   train_dataset: &train_dataset
     tokenizer:
       type: ChatGLM2Tokenizer
       vocab_file: "/path/to/tokenizer.model"
   ```

2. 执行训练命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/glm2/run_glm2_6b_lora_800T_A2_64G.yaml \
    --load_checkpoint {path}/glm2_6b.ckpt \
    --train_dataset_dir {path}/AdvertiseGen/train.json \
    --use_parallel True \
    --run_mode finetune"
   ```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 推理

MindFormers提供`GLM2-6b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡多轮推理。

```shell
# 脚本使用
bash scripts/examples/glm2/run_glm2_predict.sh CONFIG_PATH CKPT_PATH

# 参数说明
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
```

运行如下命令进行推理：

```shell
bash scripts/examples/glm2/run_glm2_predict.sh \
 configs/glm2/predict_glm2_6b.yaml \
 path/to/glm2_6b.ckpt

# 推理结果：
# 你好:
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
# 请介绍一下杭州:
# 杭州是中国浙江省省会，位于浙江省东南部，地处浙江省北部，东临东海，南接福建省，北与江苏省毗邻，是中国著名的旅游城市之一。
# 杭州有着悠久的历史和文化，被誉为“人间天堂”，被誉为“南宋都城”，是中国南方著名的历史文化名城之一。杭州还被誉为“全国最具幸福感城市”，具有丰富的历史遗存、优美的自然风光和浓郁的文化氛围。
# 杭州的经济以服务业为主导产业，特别是交通运输、仓储和邮政业。同时，杭州也是中国重要的电子商务和互联网产业基地之一，被誉为“中国电子商务之都”。
# 杭州的著名景点包括西湖、灵隐寺、千岛湖、钱塘江等。西湖是中国著名的风景名胜区之一，被誉为“人间天堂”，灵隐寺是中国著名的佛教寺庙之一，千岛湖和钱塘江是中国著名的自然风景区之一。
# 杭州还拥有丰富的人文资源，被誉为“人间天堂”的杭州西湖、灵隐寺、千岛湖、钱塘江等景点，以及宋城、南宋御街等历史文化景点，都是游客前来杭州旅游的热门景点。
# 那里有什么好吃的吗:
# 杭州是中国著名的美食城市之一，有许多特色美食和传统菜肴。以下是一些杭州的著名美食:
# 1. 西湖醋鱼：这是杭州最著名的菜肴之一，鱼肉鲜美，入口即化，佐以香醋、糖、姜丝等调料，口感酸甜适中。
# 2. 龙井虾仁：以当地特产的龙井茶为佐料，将鲜嫩的虾仁炒制而成，口感清香可口。
# 3. 灌汤包：又称小笼包，是杭州的传统点心之一。包子的皮软馅鲜，汤汁鲜美，非常受欢迎。
# 4. 姜母鸭：这是一道杭帮菜，以鸭肉、姜母、葱等调料烹制而成，口感鲜美。
# 5. 老字号小吃：杭州还有很多老字号小吃店，如胡同口烤肉串、孔府家宴、宋嫂鱼羹等，是当地居民和游客的美食选择。
# 此外，杭州还有许多特色小吃，如粽子、臭豆腐、糯米鸡、肉夹馍、鸭血粉丝汤等，让人垂涎欲滴。
```

## 评测

评测使用 `configs/glm2/run_glm2_6b_finetune_eval.yaml` 和`configs/glm2/run_glm2_6b_lora_eval.yaml`配置文件，配置文件中定义了评测所需的各配置项。

### 文本生成

评测数据集可参考[数据集下载](#数据集下载)。

配置文件修改部分如下：

```yaml
load_checkpoint: '{path}/glm2_6b.ckpt'          # 模型权重文件路径
model:
  model_config:
    seq_length: 256
eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "{path}/AdvertiseGen/dev.json" # 数据集路径
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "{path}/tokenizer.model"        # 词表路径
  max_source_length: 256
  max_target_length: 256
```

> 注：评测时模型`seq_length`需要等于评测数据集的`max_source_length`和`max_target_length`。因此修改yaml中模型`seq_length`为256。

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_finetune_eval.yaml` glm2模型推理配置，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快。

```shell
python run_mindformer.py \
 --config configs/glm2/run_glm2_6b_finetune_eval.yaml \
 --run_mode eval \
 --load_checkpoint {path}/glm2_6b_finetune.ckpt \
 --device_id 0 \
 --use_parallel False
```

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_lora_eval.yaml` glm2_lora模型推理配置，此配置可用于lora模型，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快。

```shell
python run_mindformer.py \
 --config configs/glm2/run_glm2_6b_lora_eval.yaml \
 --run_mode eval \
 --load_checkpoint {path}/glm2_6b_lora.ckpt \
 --device_id 0 \
 --use_parallel False
```

**注意**：单卡评测时，应将yaml文件中 model:model_config:batch_size 修改为等于 runner_config:batch_size

### 边训边评估

1. 使用 `Rouge-1`、`Rouge-2` 等指标评测

   使用该指标评测时速度较慢，推荐使用 `PerplexityMetric` 评测。

   将训练配置文件的 `do_eval: False` 设置为 `do_eval: True`，并且需要将 `train_dataset` 和 `eval_dataset` 的 `max_source_length`、`max_target_length` 以及 `batch_size`项设置为相同值，并且保持 `max_source_length + max_target_length + 1 = seq_length`，如下所示：

   ```yaml
   do_eval: True
   eval_step_interval: 1788
   eval_epoch_interval: -1

   metric:
     type: ADGENMetric

   model:
     model_config:
       seq_length: 192
   train_dataset: &train_dataset
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   eval_dataset: &eval_dataset
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   ```

2. 使用 `PerplexityMetric` 指标评测

   将训练配置文件的 `do_eval: False` 设置为 `do_eval: True`，并且需要将 `train_dataset` 和 `eval_dataset` 的 `max_source_length`、`max_target_length` 、`phase` 以及 `batch_size`项设置为相同值，并且保持 `max_source_length + max_target_length + 1 = seq_length`，如下所示：

   ```yaml
   do_eval: True
   eval_step_interval: 1788
   eval_epoch_interval: -1

   metric:
     type: PerplexityMetric

   model:
     model_config:
       seq_length: 192
   train_dataset: &train_dataset
     data_loader:
       phase: "train"
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   eval_dataset: &eval_dataset
     data_loader:
       phase: "train"
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   ```

mindformers通过 `eval_step_interval` 和 `eval_epoch_interval` 两项配置参数来控制边训练边评估的执行间隔，参数含义如下：

- **eval_step_interval**: 评估step间隔, 默认为100，表示每100个step间隔执行一次评估；配置为大于0的数表示每隔所配置的step数后执行一次评估，配置为小于0的数则表示禁用step评估；注意：在数据下沉模式下，step间隔值建议配置为sink size的倍数。
- **eval_epoch_interval**: 评估epoch间隔, 默认为-1，表示禁用epoch结束时的评估；配置为大于0的数表示每隔所配置的epoch数后执行一次评估，配置为小于0的数则表示禁用epoch评估；注意：数据下沉模式下，epoch所包含的step数将从数据集大小变为sink size的大小，将在 `sink_size * eval_epoch_interval` 个step后执行一次评估。
