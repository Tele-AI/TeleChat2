# Benchmark 工具说明文档

## 概述

Benchmark 工具支持对模型进行在线或离线的推理和微调。该工具提供了简化的推理和微调流程，方便用户快速对模型进行测试或应用。本说明文档将重点介绍离线推理的使用方法。其他部分将会在后续更新中补充。

## 离线推理使用指南

要在离线模式下使用 Benchmark 工具进行推理，用户需按照以下步骤进行配置和操作：

1. 准备所需文件

    请确保所有需要的文件都位于同一个文件夹下，包括以下内容

   - 模型文件

    ```text
    xxx/
      ├── xxx_tokenizer.py       # tokenizer
      ├── xxx_transformer.py     # transformer层实现
      ├── xxx_config.py          # 模型config
      └── xxx.py                 # 模型实现
    ```

   - 配置文件
   - 推理所需的其他资源文件，如`tokenizer.model`
   - 如果需要加载权重，请确保将MindFormers格式权重文件也放置在同一文件夹内

2. 修改 YAML 配置文件

   为了在推理过程中正确利用register机制注册处理组件，需要在 yaml 配置文件的`model_config`字段中增加`auto_map`字段。InternLM2配置示例如下：

   ```yaml
   model_config:
     ...
     auto_map:
       AutoModel: internlm2.InternLM2ForCausalLM
       AutoConfig: internlm2_config.InternLM2Config
       AutoTokenizer: [internlm2_tokenizer.InternLM2Tokenizer, null]
     ...
   ```

3. 启动推理脚本

   准备好所有文件并修改好配置文件后，通过以下命令启动推理脚本

   ```bash
   bash scripts/benchmark/inference_tool.sh \
       --mode MODE \
       --model_name_or_dir MODEL_NAME_OR_DIR \
       --predict_data PREDICT_DATA \
       --device DEVICE \
       --args ARGS
   ```

   参数说明
   - `--mode | -m`: 指定运行模式，选择单卡（single）或多卡（multi）模式。
   - `--model_name_or_dir | -n`: 如果使用离线模式，提供本地模型的路径；如果使用在线模式，则提供模型名称。
   - `--predict_data | -i`: 指定输入的预测数据文件。
   - `--device | -d`: 指定并行运行的卡数。
   - `--args | -a`: 指定与模型相关的额外参数。

## 示例

如使用双卡，模型存储在`/home/user/models/internlm2`路径下，预测数据为`你好，`，则命令如下：

```bash
bash scripts/benchmark/inference_tool.sh -m parallel -n /home/user/models/internlm2 -i '你好，' -d 2
```

执行该命令后，系统将根据提供的配置和输入数据，启动离线推理过程。
