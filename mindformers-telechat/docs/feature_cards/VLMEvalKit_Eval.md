# MindFormers多模态大模型评估工具

## 基本介绍

大模型的训练效果需要评测任务来作为衡量标准，而当前MindFormers暂无有效的多模态理解评测方法，对模型推理能力没有有效评估。VLMEvalKit是一款专为大型视觉语言模型评测而设计的开源工具包，支持在各种基准测试上对大型视觉语言模型进行一键评估，无需进行繁重的数据准备工作，让评估过程更加简便。

MindFormers多模态大模型评估工具遵循VLMEvalKit架构，完成了对MindFormers中视觉语言模型的适配工作，不仅可以通过评测结果对模型的训练效果进行全面的评估，还显著增强了模型评测过程的便捷性和可扩展性。

## 关键特性

本工具提供了一套简单直观的评估方法，具有以下特性：

1. 支持自动下载评测数据集；
2. 支持用户自定义输入多种数据集和模型；
3. 一键生成评测结果。

## 支持模型

本评测工具目前只适配了`cogvlm2-llama3-chat-19B`，其他模型在后续的版本中会逐渐适配。

## 环境及数据准备

### 安装VLMEvalKit

用户可以通过以下命令进行安装：

```shell
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### MindFormers环境及模型数据准备

准备MindFormers环境，准备模型数据和权重（以`cogvlm2-llama3-chat-19B`为例），参考[环境及cogvlm2数据准备](../model_cards/cogvlm2_image.md#环境及数据准备)。

## 评测

MindFormers提供`cogvlm2-llama3-chat-19B`的评测示例，目前支持单卡单batch评测。

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml`

   ```yaml
   load_checkpoint: "/{path}/cogvlm2.ckpt"  # 指定权重文件路径
   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: False                       # 关闭动态shape

     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动评测脚本

   ```shell
   export USE_ROPE_SELF_DEFINE=True
   python eval_with_vlmevalkit.py \
    --data MME \
    --model cogvlm2-llama3-chat-19B \
    --verbose \
    --work-dir /{path}/evaluate_result \
    --model-path /{path}/cogvlm2_model_path \
    --config-path /{path}/cogvlm2_config_path
   ```

   **参数说明**

     | **参数**           | **是否必选**                                       | **说明**                                                                                |
     |------------------|------------------------------------------------|---------------------------------------------------------------------------------------|
     | data       | <div style="text-align: center;">&check;</div> | 评测数据集，可以单个或多个；多个以空格分开。（例如:MME 或 MME MMBench_DEV_EN COCO_VAL）                          |
     | model     | <div style="text-align: center;">&check;</div> | 评测模型名称，可以单个或多个；多个以空格分开。（例如:cogvlm2-llama3-chat-19B 或 cogvlm2-llama3-chat-19B qwen_vl） |
     | verbose      | <div style="text-align: center;">-</div>       | 输出评测运行过程中的日志。                                                                         |
     | work-dir      | <div style="text-align: center;">-</div>       | 存放评测结果的目录。默认情况下，文件将被存储在当前目录下，且文件夹名称与模型名称相同。                                           |
     | model-path        | <div style="text-align: center;">&check;</div>                                        | 模型相关文件路径，包含模型配置文件及模型相关文件，可以单个或多个；多个以空格分开，根据评测模型数量决定，按照评测模型顺序填写。                       |
     | config-path          | <div style="text-align: center;">&check;</div>                                        | 模型配置文件路径，可以单个或多个；多个以空格分开，根据评测模型数量决定，按照评测模型顺序填写。                                       |

   > 注:
   > 1. 如果因环境限制，服务器不支持在线下载数据集，可以将本地下载好的以`.tsv`结尾的数据集文件上传至服务器`~/LMUData`目录下，完成离线评测功能。（例如：~/LMUData/MME.tsv 或 ~/LMUData/MMBench_DEV_EN.tsv 或 ~/LMUData/COCO_VAL.tsv）
   > 2. --model目前仅支持cogvlm2-llama3-chat-19B。

## 查看评估结果

按照上述方式评估后，在存储评测结果的目录中，找到以`.json`或以`.csv`结尾的文件查看评估的结果，评测结果可以和[VLMEvalKit开源评测结果](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)进行比对，来评估MindFormers模型训练的成效。

以`cogvlm2-llama3-chat-19B`模型使用COCO_VAL数据集进行评测为例：

评测结果：

```json
{
   "Bleu": [
      15.523950970070652,
      8.971141548228058,
      4.702477458554666,
      2.486860744700995
   ],
   "ROUGE_L": 15.575063213115946,
   "CIDEr": 0.01734615519604295
}
```

> 注: 评估需要对评估数据集进行全量评估，较为耗时，建议预留较长时间进行评估
