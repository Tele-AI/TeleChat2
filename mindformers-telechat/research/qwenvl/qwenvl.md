# Qwen-VL

## 模型描述

Qwen-VL 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen-VL 可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。

```text
@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```

## 模型性能

| Config                                          |             Task              |      Datasets       |   Performance   |  Phase   |
|:------------------------------------------------|:-----------------------------:|:-------------------:|:---------------:|:--------:|
| [qwenvl_9.6b](./finetune_qwenvl_9.6b_bf16.yaml) | multimodal_to_text_generation | LlaVA-Instruct-150K | 2587 tokens/s/p | Finetune |
| [qwenvl_9.6b](./predict_qwenvl_9.6b.yaml)       | multimodal_to_text_generation |          -          |   42 tokens/s   | Predict  |

## 模型文件

`Qwen-VL` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   qwenvl
     ├── qwenvl_config.py         # 配置文件
     ├── qwenvl_tokenizer.py      # tokenizer
     └── qwenvl.py                # 模型实现
   ```

2. 模型配置：

   ```text
   qwenvl
     ├── predict_qwenvl_9.6b.yaml            # qwenvl推理启动配置
     ├── finetune_qwenvl_9.6b.yaml           # qwenvl微调启动配置（2k）
     └── finetune_qwenvl_9.6b_bf16.yaml      # qwenvl微调启动配置（2k，bf16）
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwenvl
     ├── qwenvl_processor.py      # 训练和推理时候使用的数据处理
     ├── convert_weight.py        # 权重转换脚本
     ├── data_convert.py          # 数据预处理转换脚本
     └── run_qwenvl.py            # QwenVL高阶接口脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 数据集制作

目前本仓库中对Qwen-VL使用微调数据集格式同Qwen-VL开源使用数据集格式一致，如下示例：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  }
]
```

Qwen-VL开源模型中未开源相关数据集，以下提供使用公开数据集转换为上述数据格式的样例，并用于模型微调

| 数据集名称                                     |     适用模型     |   适用阶段   |                                                       下载链接                                                        |
|:------------------------------------------|:------------:|:--------:|:-----------------------------------------------------------------------------------------------------------------:|
| LlaVA-Instruct-150K detail_23k.json（对话数据） | Qwen-VL-9.6B | finetune | [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json?download=true) |
| COCO2014 Train（图片数据）                      | Qwen-VL-9.6B | finetune |                                     [Link](https://cocodataset.org/#download)                                     |

下载数据集后，需要执行`data_convert.py`脚本进行数据预处理，将原始数据转换为上述对话格式数据。

```shell
cd research/qwenvl
python data_convert.py --data_path /path/to/detail_23k.json --image_location /location/of/coco/train2014 --output_path /path/to/converted/json --user_role_name user --assistant_role_name assistant
```

其中`--data_path`表示原始对话数据路径，`--image_location`表示COCO
train2014文件夹所在路径，路径不包含train2014，`--output_path`表示转换后对话数据保存路径, `--user_role_name`
表示转换后对话中用户名称，`--assistant_role_name`表示转换后对话中助手名称。

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调/推理，用户可自行从下方链接拉取后直接使用；Base用于微调，也可进行简单推理，Chat权重可以自行通过权重转换脚本进行转换。

也可选择从HuggingFace下载所有工程文件后进行[模型权重转换](#模型权重转换)使用。

| 模型名称              |                                               MindSpore权重                                               |                  HuggingFace权重                   |
|:------------------|:-------------------------------------------------------------------------------------------------------:|:------------------------------------------------:|
| Qwen-VL-Base      | [Link](https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwenvl_base_fp16.ckpt) |   [Link](https://huggingface.co/Qwen/Qwen-VL/)   |
| tokenizer.model   |     [Link](https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwen.tiktoken)     |                        /                         |

#### 模型权重转换

进行权重转换需要安装以下依赖包。

```shell
pip install torch
pip install transformers  # 如果transformers使用tokenizers版本不是0.15.0，在权重转换完成后重装tokenizers版本为0.15.0
pip install einops transformers_stream_generator accelerate
```

执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model qwenvl --input_path /path/to/hf/dir \
--output_path /path/to/qwenvl_ms.ckpt \
--dtype fp16
```

参数说明：

`input_path`：传入从`Huggingface`下载好模型文件夹；
`output_path`：是用于存储转换后权重的路径；
`dtype`：转换权重的精度选择。

## 微调

微调阶段即Qwen-VL论文中的第三阶段，在这个阶段中，会将ViT进行冻结，仅训练QwenLM及CrossAttn部分参数，训练参数约7.78B。

### Stage-3微调

MindFormers提供了默认微调配置`finetune_qwenvl_9.6b.yaml`，默认配置中使用数据集[LlaVa-150k detail_23k](#数据集制作)
，开启LLM部分的[Flash Attention](../../docs/feature_cards/Training_Algorithms.md#flash-attention)，设置图文对话中最多包含一张图像。

#### 单机训练

1. 修改`finetune_qwenvl_9.6b_bf16.yaml`中相关配置，配置具体数据集等相关信息。

   ```yaml
   load_checkpoint: '/path/model_dir' # 权重路径，也可通过启动命令自动传入
   auto_trans_ckpt: True              # 打开自动权重转换，也可通过启动命令自动传入
   use_parallel: True
   run_mode: 'finetune'

   train_dataset: &train_dataset
     data_loader:
      type: BaseMultiModalDataLoader
      annotation_file: "/path/to/converted/json"     # 根据实际位置，填写对话json文件所在路径
      shuffle: True
     modal_to_text_transform:
        type: BaseXModalToTextTransform
        model_transform_template:
          type: QwenVLContentTransformTemplate      # QwenVL关于数据集数据处理模板
          output_columns: ["input_ids", "images", "image_context_pos", "labels"] # 文本处理后数据数据的列名，不需要配置
          mode: "train"
          dataset_dir: "/location/of/coco/train2014" # 该处配置文件夹位置与json数据集中图片路径拼接得到图片的绝对路径，如果数据集中路径已是绝对路径，该处不需要配置；当使用示例数据集时为train2014文件夹所在路径，配置项不包含train2014,
          modal_content_padding_size: 1             # 根据数据集中对话实际包含图片数量进行配置，在使用示例数据集时为1
          system_message: "You are a helpful assistant."  # 微调时，系统prompt
          user_role_name: user                            # 根据数据集转换实际配置，修改为用户角色名，默认配置为user
          assistant_role_name: assistant                  # 根据数据集转换实际配置，修改为助手角色名，默认配置为assistant
          user_prompt: ""                  # user角色prompt
          assistant_prompt: ""             # assistant角色prompt
          image_size: 448                  # 数据集加载将图片放缩至该尺寸
        max_length: 2048                   # 训练时使用seq_length
     modal_content_input_columns: [ "images"]      # 模态内容转换输入列名，该处固定为images
     modal_content_output_columns: [ "images" ]    # 模态内容转换输出列名，该处固定为images
     modal_content_transforms:                     # 模态内容转换，不需要配置，仅为示意
      - type: BatchToTensor
      - type: BatchNormalize
        mean: [ 0.48145466, 0.4578275, 0.40821073 ]
        std: [ 0.26862954, 0.26130258, 0.27577711 ]
        is_hwc: False
     net_input_columns: [ "input_ids", "images", "image_context_pos", "labels" ]  # 最终从数据集流水线中取所配置列名及其顺序送入到网络输入
     tokenizer:
       type: QwenVLTokenizer
       vocab_file: '/path/to/vocab_file'
    ```

2. 启动微调任务

运行如下命令启动单机8卡微调任务。

```shell
cd research/qwenvl
bash ../../scripts/msrun_launcher.sh "python run_qwenvl.py \
--config finetune_qwenvl_9.6b_bf16.yaml \
--run_mode finetune \
--load_checkpoint /path/to/ckpt \
--use_parallel True \
--auto_trans_ckpt True \
--vocab_file /path/to/qwen.tiktoken" 8

# 以上除config外其他传参如果在yaml文件中已经配置，可以在启动命令中不再传入
# 参数说明
# config: 配置文件路径
# run_mode: 运行模式，微调时设置为finetune
# load_checkpoint: 当使用分布式权重时传入权重文件夹路径model_dir，权重按照'model_dir/rank_0/xxx.ckpt'格式存放，传入完整权重时传入ckpt路径
# auto_trans_ckpt: 自动权重转换开关，当传入完整权重时打开
```

#### 多机训练

以Qwen-VL-9.6B进行2机16卡训练为例，只需要修改配置文件和权重即可。

1. 修改`finetune_qwenvl_9.6b_bf16.yaml`中并行相关配置，数据集配置相关可参考上文[单机训练](#单机训练)。

    ```yaml
    parallel_config:
      data_parallel: 16
      model_parallel: 1
      pipeline_stage: 1
      micro_batch_num: 1
    ```

2. 启动微调任务

   多机训练需要分别在不同节点执行命令，以下为2机16卡训练过程，参数说明以及使用更多节点参考[msrun方式启动](../../README.md#方式一使用已有脚本启动)
   多机多卡部分进行配置。

   > 注：如果各节点间使用共享存储存放工程文件，则可以使用[自动权重转换功能](../../docs/feature_cards/Transform_Ckpt.md#自动权重转换)
   ，在Qwen-VL中可通过在配置文件中设置`auto_trans_ckpt=True`或在运行命令时设置`--auto_trans_ckpt True`
   ；如果不能满足共享存储条件，需要修改配置文件`auto_trans_ckpt=False`或在运行命令时设置`--auto_trans_ckpt False`，
   此时，预训练权重可以使用[离线权重转换工具](../../docs/feature_cards/Transform_Ckpt.md#离线权重转换)
   进行转换得到切分后的分布式权重，以避免每张卡加载完整权重，导致host侧内存占用过高。

- 在节点0执行如下命令，其中192.168.1.1需要改为节点0的实际ip，将节点0作为主节点，2机共16卡且每个节点8卡。

  ```shell
  # 以使用共享盘为例
  cd research/qwenvl
  bash ../../scripts/msrun_launcher.sh "python run_qwenvl.py \
  --config finetune_qwenvl_9.6b_bf16.yaml \
  --run_mode finetune \
  --load_checkpoint /path/to/ckpt \
  --use_parallel True \
  --auto_trans_ckpt True \
  --vocab_file /path/to/qwen.tiktoken" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300
  ```

- 在节点1执行如下命令，其中192.168.1.1需要改为节点0的实际ip。

  ```shell
  cd research/qwenvl
  bash ../../scripts/msrun_launcher.sh "python run_qwenvl.py \
  --config finetune_qwenvl_9.6b_bf16.yaml \
  --run_mode finetune \
  --load_checkpoint /path/to/ckpt \
  --use_parallel True \
  --auto_trans_ckpt True \
  --vocab_file /path/to/qwen.tiktoken" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 300
  ```

## 推理

MindFormers提供`QwenVL`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理，当前多卡推理时不支持增量推理。
进行推理前，模型权重以及tokenizer文件可参考[模型权重下载](#模型权重下载)进行准备。

```shell
# 脚本使用
bash scripts/examples/qwenvl/run_qwenvl_predict.sh PARALLEL CONFIG_PATH CKPT_PATH TOKENIZER IMAGE_PATH PROMPT BATCH_SIZE DEVICE_NUM

# 参数说明
# PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
# CONFIG_PATH: 模型配置文件路径
# CKPT_PATH:   模型权重文件路径
# TOKENIZER:   模型tokenizer文件路径
# IMAGE_PATH:  推理的图片路径
# PROMPT:      对推理图片使用的Prompt
# BATCH_SIZE:  推理时使用的batch size
# DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

```shell
bash scripts/examples/qwenvl/run_qwenvl_predict.sh single \
 research/qwenvl/predict_qwenvl_9.6b.yaml \
 /path/to/qwenvl_9.6b_base.ckpt \
 /path/to/tokenizer.model \
 "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg" \
 "Describe the image in English:" \
 1 # batch_size
 # 推理结果：
 # Picture 1: <img>https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg</img>
 # Describe the image in English: A women and a dog on the bench at sunset.<|endoftext|>
```

### 多卡推理

```shell
bash scripts/examples/qwenvl/run_qwenvl_predict.sh parallel \
 research/qwenvl/predict_qwenvl_9.6b.yaml \
 path/to/qwenvl_9.6b_base.ckpt \
 path/to/tokenizer.model \
 "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg" \
 "Describe the image in English:" 1 2
 # 1 表示batch_size=1
 # 2 表示device_num=2，即使用2卡推理
```
