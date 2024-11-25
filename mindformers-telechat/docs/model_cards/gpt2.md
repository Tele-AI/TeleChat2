# GPT2

## 模型描述

GPT-2由OpenAI于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，原生GPT-2模型可分为small（124M）、medium（355M）、large（774M）、xlarge（1.5B），但在此仓中，基于GPT2扩展了13B，52B等规格。

[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

```text
@inproceedings{Radford2019LanguageMA,
  title={Language Models are Unsupervised Multitask Learners},
  author={Alec Radford and Jeff Wu and Rewon Child and David Luan and Dario Amodei and Ilya Sutskever},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:160025533}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                           |      Task       | Datasets  | SeqLength |   Performance   |  Phase   |
|:-------------------------------------------------|:---------------:|:---------:|:---------:|:---------------:|:--------:|
| [gpt2_13b](../../configs/gpt2/run_gpt2_13b.yaml) | text_generation | wikitext2 |   2048    | 1376 tokens/s/p | Finetune |
| [gpt2_13b](../../configs/gpt2/run_gpt2_13b.yaml) | text_generation | wikitext2 |   2048    |   21 tokens/s   | Predict  |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                           |      Task       | Datasets  | SeqLength |   Performance   |  Phase   |
|:-------------------------------------------------|:---------------:|:---------:|:---------:|:---------------:|:--------:|
| [gpt2_13b](../../configs/gpt2/run_gpt2_13b.yaml) | text_generation | wikitext2 |   2048    | 1286 tokens/s/p | Finetune |

## 模型文件

`gpt2`基于`mindformers`实现，主要涉及的文件有：

1、模型具体实现：

```bash
mindformers/models/gpt2
    ├── __init__.py
    ├── convert_weight.py           # 权重转换脚本
    ├── gpt2.py                     # 模型实现
    ├── gpt2_config.py              # 模型配置项
    ├── gpt2_processor.py           # gpt2预处理
    ├── gpt2_tokenizer.py           # tokenizer
    └── gpt2_modules.py             # transformer层实现
```

2、模型配置：

```bash
mindformers/configs/gpt2
    ├── finetune_gpt2_small_fp16.yaml      # gpt2 small lora低参微调配置
    ├── finetune_gpt2_small_lora_fp16.yaml      # gpt2 small lora低参微调配置
    ├── finetune_gpt2_small_txtcls_fp16.yaml    # gpt2 small文本分类模型微调配置
    ├── predict_gpt2_small_fp16.yaml           # gpt2 small模型预训练配置
    ├── pretrain_gpt2_small_fp16.yaml           # gpt2 small模型预训练配置
    └── pretrain_gpt2_13b_fp16.yaml             # gpt2 13b模型预训练配置
```

3、预处理脚本和任务启动脚本：

```bash
mindformers/tools/dataset_preprocess/gpt2
    ├── txtcls_dataset_to_mindrecord.py     # 文本分类数据集预处理
    └── wikitext2_data_process.py           # wikitext2数据集预处理
```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext2**作为预训练数据集，**alpaca**作为微调数据集。

|   数据集名称   |                                         适用模型                                         |                适用阶段                |                                                         下载链接                                                          |
|:---------:|:------------------------------------------------------------------------------------:|:----------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2 | GPT-2-small <br>GPT2-medium <br>GPT2-large <br>GPT2-xlarge <br>GPT2-13B <br>GPT2-52B | Pretrain <br>Finetune <br>Evaluate | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
|   SST-2   | GPT-2-small <br>GPT2-medium <br>GPT2-large <br>GPT2-xlarge <br>GPT2-13B <br>GPT2-52B |            <br>Evaluate            |                              [Link](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)                               |
|   IMDB    | GPT-2-small <br>GPT2-medium <br>GPT2-large <br>GPT2-xlarge <br>GPT2-13B <br>GPT2-52B |            <br>Evaluate            |               [Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)               |
|  AG-News  | GPT-2-small <br>GPT2-medium <br>GPT2-large <br>GPT2-xlarge <br>GPT2-13B <br>GPT2-52B |            <br>Evaluate            |                       [Link](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)                        |
|   COLA    | GPT-2-small <br>GPT2-medium <br>GPT2-large <br>GPT2-xlarge <br>GPT2-13B <br>GPT2-52B |            <br>Evaluate            |                                        [Link](https://nyu-mll.github.io/CoLA/)                                        |

- Wikitext2 数据预处理

    使用`mindformers/tools/dataset_preprocess/gpt2/wikitext2_data_process.py`对下载后的数据进行预处理，并生成Mindrecord数据。

    当自动下载tokenizer失败时，请手动下载 [vocab.json](https://hf-mirror.com/openai-community/gpt2/resolve/main/vocab.json?download=true)，[merges.txt](https://hf-mirror.com/openai-community/gpt2/resolve/main/merges.txt?download=true)，[config.json](https://hf-mirror.com/openai-community/gpt2/resolve/main/config.json?download=true)，[tokenizer_config.json](https://hf-mirror.com/openai-community/gpt2/resolve/main/tokenizer_config.json?download=true) 并将他们放在同一目录下，并将tokenizer路径指向此目录。

    ```bash
    python wikitext2_data_process.py \
    --input_file ./wikitext-2/wiki.train.tokens \
    --output_file ./wikitext-2.train.mindrecord \
    --max_length 1025 \
    --tokenizer_type path/to/tokenizer

    # 参数说明
    input_file:      输入下载后wiki.train.tokens的文件路径
    output_file:     输出文件的保存路径
    max_length:      最大序列长度
    tokenizer_type:  tokennizer所在文件夹, 自动下载失败时请指定此参数
    ```

    > 注: 除使用`configs/gpt2/finetune_gpt2_small_txtcls_fp16.yaml`配置文件外，预训练或者微调时，数据需处理为`configs/gpt2/*_gpt2_*_*.yaml`中`model.model_config.seq_length`的值加1，如下，当使用`pretrain_gpt2_small_fp16.yaml`配置文件执行训练时，`max_length`需设为1025。

- SST-2/IMDB/AG-News/COLA 数据预处理

    因评测前需要微调模型，所以需要生成训练/评测数据集。使用`mindformers/tools/dataset_preprocess/gpt2/txtcls_dataset_to_mindrecord.py`对下载后的数据进行预处理，并生成Mindrecord数据。

    ```bash
    python txtcls_dataset_to_mindrecord.py --dataset_name {select one from ['cola', 'sst_2', 'ag_news', 'imdb']}
                                     --input_file {your_path/train.tsv} \
                                     --output_file {your_path/dataset_name.train.mindrecord}

    python txtcls_dataset_to_mindrecord.py --dataset_name {the same as above}
                                     --input_file {your_path/dev.tsv} \
                                     --output_file {your_path/dataset_name.dev.mindrecord}
    ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过模型权重转换后进行使用。

|       模型名称       |                                                   MindSpore 权重                                                    |                     HuggingFace 权重                     |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------:|
|    gpt2_small    |     [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2.ckpt)     |   [Link](https://huggingface.co/openai-community/gpt2/tree/main)   |
| gpt2_small_lora  |  [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_lora.ckpt)   |                                 \                                  |
|     gpt2_13b     |   [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_13b.ckpt)   | [Link](https://huggingface.co/cerebras/Cerebras-GPT-13B/tree/main) |
|     gpt2_xl      |   [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_xl.ckpt)    | [Link](https://huggingface.co/openai-community/gpt2-xl/tree/main)  |
|   gpt2_xl_lora   | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_xl_lora.ckpt) |                                 \                                  |

> 注：13b的权重需要将上述链接下的`pytorch_model-00001-of-00002.bin`、`pytorch_model-00002-of-00002.bin`、`pytorch_model.bin.index.json
`、`config.json`下载并存到一个文件夹`torch_weights`中，然后使用如下命令将Huggingface的权重进行合并
> ```python
> from transformers import AutoModelForCausalLM
> model = AutoModelForCausalLM.from_pretrained("torch_weights")
> model.save_pretrained("gpt_13b.bin", max_shard_size="60GB")
> ```

#### 模型权重转换

执行`mindformers/models/gpt2/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```bash
python convert_weight.py \
    --layers LAYERS\
    --torch_path TORCH_PATH \
    --mindspore_path MINDSPORE_PATH

    # 参数说明
    layers:          模型层数
    torch_path:      下载HuggingFace权重的文件路径
    mindspore_path:  转换后的MindSpore权重文件保存路径
```

## 预训练

MindFormers提供`gpt2-small`的预训练示例，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机训练

执行msrun启动脚本，进行8卡分布式训练。各个参数位置含义参见[msrun快速启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)。

```bash
# dataset_dir可指定文件目录或文件路径，
# 指定文件路径时，读取单文件，
# 指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件
bash scripts/msrun_launcher.sh "root/mindformers/run_mindformer.py \
  --run_mode train \
  --train_dataset_dir path/to/wikitext-2.train.mindrecord \
  --config configs/gpt2/pretrain_gpt2_small_fp16.yaml \
  --use_parallel True" 8
```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage

## 微调

MindFormers提供`gpt2-small`的微调示例，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

当前模型已支持使用***Flash Attention***算法进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单机训练

执行msrun启动脚本，进行8卡分布式训练。各个参数位置含义参见[msrun快速启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)。

```bash
# dataset_dir可指定文件目录或文件路径，
# 指定文件路径时，读取单文件，
# 指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件
bash scripts/msrun_launcher.sh "root/mindformers/run_mindformer.py \
  --run_mode finetune \
  --train_dataset_dir path/to/wikitext-2.train.mindrecord \
  --config configs/gpt2/finetune_gpt2_small_fp16.yaml \
  --use_parallel True" 8
```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage

### LoRA微调

使用LoRA低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，使大模型在少量资源的情况下也能训练。MindFormers提供`gpt2-small`的LoRA微调示例，微调过程中使用的数据集可以参考[数据集下载](#数据集下载)获得。

#### 单机训练

```bash
# dataset_dir可指定文件目录或文件路径，
# 指定文件路径时，读取单文件，
# 指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件
bash scripts/msrun_launcher.sh "root/mindformers/run_mindformer.py \
    --run_mode finetune \
    --train_dataset_dir path/to/wikitext-2.train.mindrecord
    --config configs/gpt2/finetune_gpt2_small_lora_fp16.yaml \
    --load_checkpoint "the path of pretrained ckpt or gpt2" \
    --use_parallel True" 8
```

> 注：命令中的卡数须等于data_parallel\*model_parallel\*pipeline_stage

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。 涉及到模型权重的单卡或多卡转换，详细教程请参考特性文档模型[权重切分与合并](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)。

1. 获取模型切分策略文件:

    在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并:

    ```bash
    python transform_ckpt.py \
        --src_ckpt_strategy {path}/output/strategy/ \
        --src_ckpt_dir {path}/output/checkpoint/ \
        --dst_ckpt_dir {path}/target_checkpoint/ \
        --prefix gpt2

        # 参数说明
        src_ckpt_strategy: 切分策略文件路径
        src_ckpt_dir:      原切分权重文件夹
        dst_ckpt_dir:      目标路径
        prefix:            ckpt文件前缀
    ```

    > 注: `transform_checkpoints`接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以通过mindspore 2.0的cpu版本以执行该脚本。

## 推理

MindFormers提供`gpt2-small`的快速推理脚本，脚本主要通过`generate`高阶接口实现，支持单卡以及多batch推理。

```bash
# 脚本使用
bash scripts/examples/gpt2/run_gpt2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM TOKENIZER_PATH

# 参数说明
PARALLEL:        是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理(暂只支持single)
CONFIG_PATH:     模型配置文件路径
CKPT_PATH:       模型权重文件路径
DEVICE_NUM:      使用卡数, 仅开启多卡推理时生效
TOKENIZER_PATH:  Tokenizer模型路径
```

### 单卡推理

```bash
bash scripts/examples/gpt2/run_gpt2_predict.sh \
    single \
    configs/gpt2/predict_gpt2_small_fp16.yaml \
    path/to/gpt2_small.ckpt \
    1 \
    path/to/tokenizer.model
    # 多batch输出
    # <s>I love Beijing, because it's a beautiful city. It's a beautiful city. It's a ...
    # <s>GPT2 is a new class of microprocessor that is designed to be used in a wide range of applications. It is designed to be used in a wide range of applications.
    #
    # The new microprocessor is designed to be used in a wide range of applications. It is designed to be used in a wide range of applications.
    #
    # The new microprocessor ...
    #
    # <s>Huawei is a company that has been around for a long time. It's been around for a long time, and it's been around for a long time. It's been around for a long time, and it's ...
```

## 评测

以`gpt2-samll`为例，当前支持使用based model（初始权重）进行评测任务如下：

| 任务类型  |     评测指标     |           数据集           |
|:-----:|:------------:|:-----------------------:|
| 文本生成  |  Perplexity  |        WikiText2        |
| 文本分类  |     ACC      | SST-2/IMDB/AG-News/COLA |

> 注: 数据处理脚本的`max_length`入参默认是`pretrain_gpt2_small_fp16.yaml`中的`seq_length`，即`1024`。如更换使用模型，需设置数据处理脚本的`max_length`为对应yaml文件中的`seq_length`。**

### 文本生成

1. 获取数据集

    文本生成任务评测使用WikiText2数据集，可通过[数据集下载](#数据集下载)得到，并进行相应的预处理。

2. 修改模型配置文件`configs/gpt2/pretrain_gpt2_small_fp16.yaml`

    ```yaml
    metric:
      type: PerplexityMetric

    model:
      model_config:
        use_flash_attention: True
    ```

3. 执行评测命令

    ```bash
    python run_mindformer.py --config configs/gpt2/pretrain_gpt2_small_fp16.yaml \
                             --eval_dataset_dir {your_path/wikitext-2.valid.mindrecord} \
                             --run_mode eval \
                             --epochs 1
    # gpt2: PerplexityMetric: {'PerplexityMetric': {'loss': 3.24, 'PPL': 25.55}
    # gpt2_13b(需替换yaml文件): PerplexityMetric: {'PerplexityMetric': {'loss': 2.35, 'PPL': 10.49}
    ```

### 文本分类

1. 获取数据集

- [SST-2数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)数据集包含电影评论中的句子和它们情感的人类注释。类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）

- [IMDB数据集](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)影评数据集，包含5万条IMDB影评，评论的情绪是二元的，专门用于情绪分析。

- [AG-News数据集](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)数据集包含496,835条来自AG新闻语料库4大类别超过2000个新闻源的新闻文章。

- [COLA数据集](https://nyu-mll.github.io/CoLA/)数据集来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。

2. 处理数据成mindrecord格式

    数据处理文件`txtcls_dataset_to_mindrecord.py`在目录`mindformers/tools/dataset_preprocess/gpt2`下。

    ```bash
    python txtcls_dataset_to_mindrecord.py --dataset_name {select one from ['cola', 'sst_2', 'ag_news', 'imdb']}
                                           --input_file {your_path/train.tsv} \
                                           --output_file {your_path/dataset_name.train.mindrecord}

    python txtcls_dataset_to_mindrecord.py --dataset_name {the same as above}
                                           --input_file {your_path/dev.tsv} \
                                           --output_file {your_path/dataset_name.dev.mindrecord}
    ```

3. 开启微调

    因为原始权重中不包含隐向量向类别映射的参数，所以无法进行zero-shot，评测前需要事先进行微调。

    ```bash
    # 运行前请确保finetune_gpt2_small_txtcls_fp16.yaml中的model.model_config.num_labels准确，具体的，
    # sst2/cola/imdb: num_labels = 2, agnews: num_labels = 4
    python run_mindformer.py --config configs/gpt2/finetune_gpt2_small_txtcls_fp16.yaml \
                           --train_dataset_dir {your_path/dataset_name.train.mindrecord} \
                           --load_checkpoint {the path of pretrained ckpt} \
                           --run_mode finetune
    ```

4. 开启评测

    ```bash
    # 运行前请确保finetune_gpt2_small_txtcls_fp16.yaml中的model.model_config.num_labels准确，具体的，
    # sst2/cola/imdb: num_labels = 2, agnews: num_labels = 4
    python run_mindformer.py --config configs/gpt2/finetune_gpt2_small_txtcls_fp16.yaml \
                           --eval_dataset_dir {your_path/dataset_name.dev.mindrecord} \
                           --run_mode eval \
                           --epochs 1
    # ACC: COLA-0.693, SST-2-0.908, IMDB-0.934, AG-News-0.941
    ```