# 使用datasets进行数据集预处理和加载

## 目的

接入魔乐仓库、HuggingFace仓库，在线加载数据集，扩大数据集来源。

使用dataset增强数据集加载和处理能力。

## 对接HuggingFace开源社区

1、环境准备

环境变量 `HF_ENDPOINT`可以控制开源社区huggingFace实际使用的远程仓库，未配置时默认为 `https://huggingFace.co`，针对国内环境，需要配置成镜像地址 `https://hf-mirror.com`

2、安装依赖

```shell
git clone https://gitee.com/openmind-ai/openmind-hub.git
cd openmind-hub
pip install -e .
cd ..
pip install datasets==2.18.0
git clone https://gitee.com/openmind-ai/openmind-extension-for-datasets.git
cd openmind-extension-for-datasets
pip install -e .
cd ..
```

## 对接魔乐开源社区

1、环境准备

环境变量 `OPENMIND_HUB_ENDPOINT`可以控制魔乐开源社区实际使用的远程仓库，未配置时默认为 `https://telecom.openmind.cn`。

2、安装依赖

```shell
git clone https://gitee.com/openmind-ai/openmind-hub.git
cd openmind-hub
pip install -e .
cd ..
pip install datasets==2.18.0
git clone https://gitee.com/foundation-models/openmind-datasets.git
cd openmind-datasets
pip install -e .
cd ..
```

3、注意事项

当环境安装了openmind-datasets三方件时，默认对接的是魔乐开源社区，如果这是想对接HuggingFace，环境变量`USE_OM`可以控制具体对接哪个社区，默认值为`ON`为魔乐社区，修改为`OFF`对接HuggingFace社区

## CommonDataLoader方式加载数据集

### 功能

CommonDataLoader定义了通用的流程步骤：1、加载远端数据集（支撑huggingFace、魔乐社区）得到开源的datasets数据集；2、自定义数据处理DataHandler模块（可选：支持用户对加载到的数据集做定制逻辑转换）；3、开源的datasets转换为ms.datasets

### 参数

加载远端数据集使用的时huggingFace提供datasets三方件`dataset.load_dataset()`方法，用户可以使用所有load_dataset支持的参数进行传递，具体使用指导可以参考`https://huggingface.co/docs/datasets/package_reference/loading_methods` 对应的国内镜像地址`https://hf-mirror.com/docs/datasets/package_reference/loading_methods` 以下是一些常见字段的说明

- type: 必填，数据加载的处理方式，支持3种方式：MindDataset、CommonDataLoader、自定义XXXDataLoader
- path： 必填，对接远端数据集路径，
- shuffle: 必填，数据集是否打乱
- handler：可选，自定义数据处理，配套type为CommonDataLoader时使用
- input_columns：可选，datasets转换为ms.datasets时，使用哪些字段转换，默认为`["input_ids", "labels"]`

### 自定义datahandler

用户可以使用自定义的dataHandler逻辑，对加载到的远端数据集进行数据预处理定制逻辑

#### 参数

- type: 必填，自定义数据处理handler名称，自定义handler必须继承BaseInstructDataHandler
- tokenizer_name: 必填，使用的tokenizer分词器名称
- seq_length: 必填，序列长度
- output_columns: 必填，数据预处理后输出的数据列
- prompt_key: 可选，增加prompt处理后数据列名称
- tokenizer: 可选，tokenizer配置参数, 可以是字典或者字符串，也可以直接配置tokenizer对象。

#### 开发样例

自定义dataHandler一般放在mindformers/dataset/handler目录下，自定义的需要继承抽象基类base_handler，需要实现format_func、tokenize_func两个方法，可以参考alpaca_handler.py

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AlpacaInstructDataHandler(BaseInstructDataHandler):
```

format_func用于实现如何从原始数据中，转换成你所需要的数据格式，

```python
def format_func(self, data):
    # 自定义处理逻辑
```

tokenize_func方法用于把处理后的数据进行按自定义分词

```python
def tokenize_func(self, messages):
  # 自定义处理逻辑
```

## alpaca数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件 `finetune_llama2_7b.yaml`。

修改如下参数：

```yaml
train_dataset:
  input_columns: &input_columns ["input_ids", "labels"]
  data_loader:
    type: CommonDataLoader
    shuffle: True
    split: "train"
    path: "AI_Connect/alpaca"
    input_columns: *input_columns
    handler:
      type: AlpacaInstructDataHandler
      tokenizer_name: llama2_13b
      seq_length: 4096
      prompt_key: "conversations"
      output_columns: *input_columns

# 参数说明
input_columns: 必填，输入的数据的列名
data_loader.type: 必填，数据加载处理的类名
data_loader.shuffle: 必填，数据集是否打乱
data_loader.path: 可选，加载数据集的远端路径
data_loader.input_columns：可选，datasets转换为ms.datasets时，使用哪些字段转换，默认为["input_ids", "labels"]
data_loader.handler: 可选，数据预处理类配置，为空时不做数据处理
data_loader.handler.type: 数据预处理类的类名
data_loader.handler.tokenizer_name: 分词器名称
data_loader.handler.seq_length: 序列长度
data_loader.handler.prompt_key: 可选，增加prompt处理后数据列名称
data_loader.handler.output_columns: 可选，数据预处理后输出的数据列
```

## ADGEN数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件 `run_glm3_6b_finetune_2k_800T_A2_64G.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    path: "xxx/ADGEN"
    split: "train"
    shuffle: True
    input_columns: ["prompt", "answer"]
    handler:
      type: AdgenInstructDataHandler
      output_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM3Tokenizer
    vocab_file: "/data/z00827078/GLM3/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 1024
  max_target_length: 1023
  ignore_pad_token_for_loss: True
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 8
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  phase: "train"
  version: 3
  seed: 0

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.path: 必填，加载数据集路径
data_loader.shuffle: 必填，数据集是否打乱
data_loader.split: 可选，数据集子集，默认加载train集
data_loader.input_columns：可选，datasets转换为ms.datasets时，使用哪些字段转换，默认为["input_ids", "labels"]
data_loader.handler: 可选，自定义数据处理器
data_loader.handler.type: 可选，自定义数据处理器类型名称
data_loader.handler.output_columns: 可选，处理完后输出的数据集列名
```

## Qwen-VL数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件 `finetune_qwenvl_9.6b_bf16.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    path: "xxx"
    input_columns: ["conversations"]
    shuffle: False
    handler:
      type: LlavaInstructDataHandler
      image_dir: "xxxx"
      output_columns: ["conversations"]

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.path: 必填，加载数据集路径
data_loader.shuffle: 必填，数据集是否打乱
data_loader.input_columns: 可选，datasets转换为ms.datasets时，使用哪些字段转换，默认为["input_ids", "labels"]
data_loader.handler: 可选，自定义数据处理器
data_loader.handler.type: 可选，自定义数据处理器类型名称
data_loader.handler.image_dir: 必填，图片目录路径
data_loader.handler.output_columns: 可选，处理完后输出的数据集列名
```

## code_alpaca(DeepSeek-Coder)数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件 `finetune_deepseek_33b.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    shuffle: False
    path: 'AI_Connect/CodeAlpaca-20K'
    split: 'train'
    handler:
      type: CodeAlpacaInstructDataHandler
      tokenizer_name: ''
      seq_length: 4096
      prompt_key: "conversations"
      output_columns: ["input_ids", "labels"]
      tokenizer:
        unk_token: None
        bos_token: '<｜begin▁of▁sentence｜>'
        eos_token: '<|EOT|>'
        pad_token: '<｜end▁of▁sentence｜>'
        vocab_file: None
        tokenizer_file: "path/to/deepseek/tokenizer.json"  # tokenizer.json
        type: LlamaTokenizerFast

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.path: 必填，加载数据集的远端路径
data_loader.shuffle: 必填，数据集是否打乱
data_loader.split: 可选，数据集子集，默认加载train集
data_loader.handler: 可选，数据预处理类配置，为空时不做数据处理
data_loader.handler.type: 数据预处理类的类名
data_loader.handler.tokenizer_name: 分词器名称
data_loader.handler.seq_length: 序列长度
data_loader.handler.prompt_key: 可选，增加prompt处理后数据列名称
data_loader.handler.tokenizer: 必填，tokenizer的详细配置
```
