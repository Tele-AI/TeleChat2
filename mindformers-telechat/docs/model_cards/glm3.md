# ChatGLM3

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：**更强大的基础模型**，**更完整的功能支持**，**更全面的开源序列**

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                   |      Task       | SeqLength | Datasets |   Performance   |  Phase   |
|:---------------------------------------------------------|:---------------:|:---------:|:--------:|:---------------:|:--------:|
| [glm3_6b](../../configs/glm3/finetune_glm3_6b_bf16.yaml) | text_generation |   2048    |  ADGEN   | 3450 tokens/s/p | Finetune |
| [glm3_6b](../../configs/glm3/predict_glm3_6b.yaml)       | text_generation |   2048    |    /     |  627 tokens/s   | Predict  |

## 模型文件

`chatGLM3-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm2      # glm3模型继承glm2的代码
    ├── __init__.py
    ├── convert_weight.py          # huggingface权重转ckpt实现
    ├── glm2.py                    # 模型实现
    ├── glm2_config.py             # 模型配置项
    ├── glm2_modules.py            # 模组实现
    ├── glm3_tokenizer.py          # tokenizer
    └── glm2_transformer.py        # transformer层实现
    ```

2. 模型配置：

    ```text
    configs/glm3
    ├── predict_glm3_6b.yaml                              # 在线推理配置文件
    ├── run_glm3_6b_finetune_2k_800T_A2_64G.yaml          # Atlas 800T A2 最佳性能全量微调启动配置
    ├── run_glm3_6b_finetune_800T_A2_64G.yaml             # Atlas 800T A2 ADGEN 全量微调启动配置
    ├── run_glm3_6b_multiturn_finetune_800T_A2_64G.yaml   # Atlas 800T A2 多轮对话全量微调启动配置
    └── run_glm3_6b.yaml                                  # ChatGLM3配置模板
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供[`ADGEN`](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集和作为`ToolAlpaca`数据集作为微调数据集，`ADGEN`数据集无需处理即可使用。

| 数据集名称      |    适用模型     |   适用阶段   |                                                                                  下载链接                                                                                  |
|:-----------|:-----------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ADGEN      | ChatGLM3-6b | Finetune |                                                   [Link](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)                                                   |
| ToolAlpaca | ChatGLM3-6b | Finetune | [Source](https://github.com/tangqiaoyu/ToolAlpaca) / [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tool_alpaca.jsonl) |

- **ToolAlpaca 数据预处理**

用户可以通过[Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tool_alpaca.jsonl)下载处理好的数据集使用，
也可以通过[Source](https://github.com/tangqiaoyu/ToolAlpaca)下载原始数据，通过如下处理后使用：

1. 下载数据处理脚本[format_tool_alpaca.py](https://github.com/THUDM/ChatGLM3/blob/7cd5bc78bd6232d02764b60b33874bb2d63a0df0/finetune_chatmodel_demo/scripts/format_tool_alpaca.py)

2. 执行数据处理脚本，在执行路径生成处理好的数据`tool_alpaca.jsonl`

   ```shell
   python format_tool_alpaca.py --path ToolAlpaca/data/train_data.json
   ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenizer.model)

| 模型名称        | MindSpore权重 |                  HuggingFace权重                   |
|:------------|:-----------:|:------------------------------------------------:|
| ChatGLM3-6b |      /      | [Link](https://huggingface.co/THUDM/chatglm3-6b) |

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

MindFormers提供`ChatGLM3-6B`的微调示例， 过程中使用`ADGEN`数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

> 注：微调时模型的`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。 在配置文件中默认的`seq_length: 192`以及`max_source_length: 256`和`max_target_length: 256`适用于ADGEN数据集，
> 对于其他数据集，可以将数据集转换为`token_id`，使`seq_length`等于`token_id`的最大长度，`seq_length`太大影响训练性能，太小影响训练精度，需要做出权衡。

1. 修改`configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml`配置文件：

   ```yaml
   train_dataset: &train_dataset
     tokenizer:
       type: ChatGLM3Tokenizer
       vocab_file: "/path/to/tokenizer.model"
   ```

2. 执行分布式训练命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml \
    --load_checkpoint {path}/glm3_6b.ckpt \
    --train_dataset_dir {path}/AdvertiseGen/train.json \
    --use_parallel True \
    --run_mode finetune"
   ```

#### 多机训练

`ChatGLM3-6B`多机多卡训练可以参考[多机多卡启动方式](../../README.md#多机多卡)。

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 推理

MindFormers提供`GLM3-6b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡推理以及多角色推理。

```shell
# 脚本使用
bash bash scripts/examples/glm3/run_glm3_predict.sh PARALLEL CONFIG_PATH CKPT_PATH

# 参数说明
PARALLEL:    推理模式选择, 'single'表示单卡推理, 'multirole'表示多角色推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
```

### 单卡推理

运行如下命令进行单卡推理：

```shell
bash scripts/examples/glm3/run_glm3_predict.sh single \
 configs/glm3/predict_glm3_6b.yaml \
 path/to/glm3_6b.ckpt

# 输出推理结果
# 你好:
# 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
# 请介绍一下华为:
# 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、
# 云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。
# ...
```

### 多角色推理

运行如下命令进行多角色推理：

```shell
bash scripts/examples/glm3/run_glm3_predict.sh multirole \
 configs/glm3/predict_glm3_6b.yaml \
 path/to/glm3_6b.ckpt

# 输入prompt: 假设你现在是一个导游，请尽可能贴近这个角色回答问题。
# 输出推理结果
# 您好，我是您的人工智能助手，也可以是你的导游。请问有什么问题我可以帮您解答呢？
# 我打算1月份去海南玩，可以介绍一下海南有哪些好玩的，好吃的么？
# 当然可以！海南是一个风景优美、气候宜人的热带海洋省份，拥有丰富的旅游资源和美食。以下是一些您可能会感兴趣的景点和美食：
# ...
```

## 常见问题

### 网络训练 loss 不下降、网络训练溢出、`overflow_cond=True`

执行训练前设置环境变量：

```shell
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

重新启动训练。

### 推理速度非常慢、Mindspore只能跑在CPU上、报错中含有 `te`、`tbe`、`tvm`等字样

一般是 Mindspore + Ascend 环境安装问题，确认环境安装过程参照[安装指南](https://www.mindspore.cn/install)并且成功设置了环境变量。执行：

```shell
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```

假如执行输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

并且没有报错，则说明成功安装了环境。

或许你想问，有没有更方便的环境安装方式？恭喜你，有的，我们还提供现成的
[docker镜像](http://mirrors.cn-central-221.ovaijisuan.com/mirrors.html)，可以依据需求自行取用。

### 微调报错：Sync stream Failed、exec graph xxx failed

这类报错较为宽泛，可以打开昇腾host日志进一步定位。

```shell
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

打开昇腾host日志后模型性能将明显下降，定位问题结束后需要取消昇腾日志：

```shell
unset ASCEND_GLOBAL_EVENT_ENABLE ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT
```

### 微调报错：the strategy is xxxxxx, shape xxxx cannot be divisible by value x

检查模型句长是否满足 `max_source_length + max_target_length + 1 = seq_length` 的要求。
