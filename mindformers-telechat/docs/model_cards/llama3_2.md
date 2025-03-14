# Llama 3.2

## 模型描述

Llama 3.2多语言大模型集合是1B和3B大小的预训练和指令调整生成模型的集合。Llama 3.2指令调整的纯文本模型针对多语言对话用例进行了优化，
包括代理检索和摘要任务。在常见的行业基准上，它们的性能优于许多可用的开源和封闭式聊天模型。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                      |      Task       | Datasets | SeqLength | Performance  |  Phase  |
|:--------------------------------------------|:---------------:|:--------:|:---------:|:------------:|:-------:|
| [llama3_2_3b](./predict_llama3_2_3b.yaml)   | text_generation |    -     |   4096    | 1643 tokens/s | Predict |

## 模型文件

`Llama 3.2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：

   ```text
   research/llama3_2
       └── predict_llama3_2_3b.yaml    # 3B推理配置
   ```

3. 数据预处理脚本和任务启动脚本：

   ```text
   research/llama3_2
       └── llama3_2_tokenizer.py     # llama3_2 tokenizer处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)
和[版本匹配关系](../../README.md#版本匹配关系)。

|      模型      |      硬件       | 全量微调 | 推理 |
|:------------:|:-------------:|:----:|:--:|
| Llama3.2-3b  | Atlas 800T A2 | 单节点 | 单卡 |

### 数据集及权重准备

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Llama-3.2-3B)

| 模型名称         | MindSpore权重 |                        HuggingFace权重                         |
|:-------------|:-----------:|:------------------------------------------------------------:|
| llama3_2-3B  |      \      | [Link](https://huggingface.co/meta-llama/Llama-3.2-3B)  |

> 注: 请自行申请huggingface上llama3_2使用权限，并安装transformers=4.40版本

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 推理

MindFormers提供`llama3_2-3b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡推理和多卡推理。推理输入默认不添加bos字符，如果需要添加可在模型的yaml文件中增加add_bos_token选项。

```shell
# 脚本使用
bash scripts/examples/llama3/run_llama3_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
VOCAB_FILE:  词表路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

以`llama3_2-3b`单卡推理为例。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh single \
 research/llama3_2/predict_llama3_2_3b.yaml \
 path/to/llama3_2_3b.ckpt \
 path/to/tokenizer.model
```

### 多卡推理

以`llama3_2-3b`4卡推理为例。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh parallel \
 research/llama3_2/predict_llama3_2_3b.yaml \
 path/to/llama3_2_3b.ckpt \
 path/to/tokenizer.model 4
```

## 基于MindIE的服务化推理

MindIE，全称Mind Inference Engine，是华为昇腾针对AI全场景业务的推理加速套件。

MindFormers承载在模型应用层MindIE-LLM中，MindIE-LLM是大语言模型推理框架，提供API支持大模型推理能力。

MindIE安装流程请参考[MindIE服务化部署文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/mindie_deployment.html)。

以下例子默认已完成MindIE安装部署且仅适用于**MindIE RC3版本**，且安装路径均为默认路径`/usr/local/Ascend/`。

### 单卡推理

此例子使用llama3_2-3B模型演示。

#### 修改MindIE启动配置

打开mindie-service中的config.json文件，修改server相关配置。

```bash
vim /usr/local/Ascend/mindie/1.0.RC3/mindie-service/conf/config.json
```

需要关注以下字段的配置

1. `ModelDeployConfig.ModelConfig.backendType`

   该配置为对应的后端类型，必填"ms"。

   ```json
   "backendType": "ms"
   ```

   2. `ModelDeployConfig.ModelConfig.modelWeightPath`

      该配置为模型配置文件目录，放置模型和tokenizer等相关文件。

      以llama3_2-3B为例，`modelWeightPath`的组织结构如下：

      ```reStructuredText
      mf_model
       └── llama3_2_3b
              ├── config.json                             # 模型json配置文件
              ├── tokenizer.model                         # 模型vocab文件，hf上对应模型下载
              ├── predict_llama3_2_3b.yaml                # 模型yaml配置文件
              ├── llama3_2_tokenizer.py                   # 模型tokenizer文件,从mindformers仓中research目录下找到对应模型复制
              └── llama3_2_3b.ckpt                        # 单卡模型权重文件
      ```

      predict_llama3_2_3b.yaml需要关注以下配置：

      ```yaml
      load_checkpoint: '/mf_model/llama3_2_3b/llama3_2_3b.ckpt' # 为存放模型单卡权重文件路径
      use_parallel: False
      model:
        model_config:
          type: LlamaConfig
          auto_map:
            AutoTokenizer: [llama3_2_tokenizer.Llama3Tokenizer, null]
      processor:
        tokenizer:
          vocab_file: "/mf_model/llama3_2_3b/tokenizer.model"  #vocab文件路径
      ```

      模型的config.json文件可以使用`save_pretrained`接口生成，示例如下：

      ```python
      from mindformers import AutoConfig

      model_config = AutoConfig.from_pretrained("/mf_model/llama3_2_3b/predict_llama3_2_3b.yaml ")
      model_config.save_pretrained(save_directory="/mf_model/llama3_2_3b", save_json=True)
      ```

      模型权重下载和转换可参考 [权重格式转换指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)。

      准备好模型配置目录后，设置参数`modelWeightPath`为该目录路径。

```json
   "modelWeightPath": "/mf_model/llama3_2_3b"
```

最终修改完后的config.json如下：

```json
{
    "Version": "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindservice.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress": "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrl" : "security/certs/server_crl.pem",
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrl" : "security/certs/management/server_crl.pem",
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "pdInterNodeTLSEnabled": false,
        "pdCommunicationPort": 1121,
        "interNodeTlsCaFile" : "security/grpc/ca/ca.pem",
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrl" : "security/certs/server_crl.pem",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb"
    },

    "BackendConfig": {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled": false,
        "multiNodesInferPort": 1120,
        "interNodeTLSEnabled": true,
        "interNodeTlsCaFile": "security/grpc/ca/ca.pem",
        "interNodeTlsCert": "security/grpc/certs/server.pem",
        "interNodeTlsPk": "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrl" : "security/grpc/certs/server_crl.pem",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
        "ModelDeployConfig":
        {
            "maxSeqLen" : 8192,
            "maxInputTokenLen" : 8192,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType": "Standard",
                    "modelName" : "llama3_2_3b",
                    "modelWeightPath" : "/mf_model/llama3_2_3b",
                    "worldSize" : 1,
                    "cpuMemSize" : 30,
                    "npuMemSize" : 25,
                    "backendType": "ms"
                }
            ]
        },

        "ScheduleConfig":
        {
            "templateType": "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 16,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 30,
            "maxIterTimes" : 4096,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : true,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

> 注：为便于测试，`httpsEnabled`参数设置为`false`，忽略后续https通信相关参数。

#### 启动服务

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

打印如下信息，启动成功。

```json
Daemon start success!
```

#### 请求测试

服务启动成功后，可使用curl命令发送请求验证，样例如下：

```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请介绍一下自己<|im_end|>\n<|im_start|>assistant\n","stream": false}' http://127.0.0.1:1025/generate
```

返回推理结果验证成功：

```json
{"generated_text":"我是系统的助理<|im_end|>\n<"}
```
