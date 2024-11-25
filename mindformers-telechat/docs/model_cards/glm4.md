# GLM-4

## 模型描述

GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B
及其人类偏好对齐的版本 GLM-4-9B-Chat 均表现出较高的性能。 除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function
Call）和长文本推理（支持最大 128K 上下文）等高级功能。 本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持
1M 上下文长度（约 200 万中文字符）的模型。

```text
@article{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools},
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                   |      Task       | Datasets | SeqLength |  Phase  | Performance  |
|:---------------------------------------------------------|:---------------:|:--------:|:---------:|:-------:|:------------:|
| [GLM-4-9B](../../configs/glm4/predict_glm4_9b_chat.yaml) | text_generation |    -     |   8192    | Predict | 256 tokens/s |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                               |      Task       | Datasets | SeqLength |  Phase   |   Performance   |
|:-----------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:---------------:|
| [GLM-4-9B](../../configs/glm4/finetune_glm4_9b.yaml) | text_generation |  alpaca  |   8192    | Finetune | 2339 tokens/s/p |

## 模型文件

`GLM-4-9B-Chat`、`GLM-4-9B`  基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm2            # glm4复用glm2的代码实现
        ├── __init__.py
        ├── convert_weight.py          # huggingface权重转ckpt实现
        ├── glm2.py                    # 模型实现
        ├── glm2_config.py             # 模型配置项
        ├── glm2_modules.py            # 模组实现
        ├── glm4_tokenizer.py          # tokenizer
        └── glm2_transformer.py        # transformer层实现
    ```

2. 模型配置：

    ```text
    configs/glm4
        ├── predict_glm4_9b_chat.yaml        # Atlas 800T A2推理配置
        └── finetune_glm4_9b.yaml            # Atlas 800T A2微调配置
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)
和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供`alpaca`数据集示例处理脚本制作[微调](#微调)示例数据集。

| 数据集名称        |  适用模型   |   适用阶段   |                                            下载链接                                            |
|:-------------|:-------:|:--------:|:------------------------------------------------------------------------------------------:|
| alpaca       | glm4-9b | Finetune |      [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)       |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **alpaca 数据预处理**

  执行`mindformers/tools/dataset_preprocess/glm4/alpaca_converter.py`，将原始数据集转换为jsonl格式。

  ```shell
  python mindformers/tools/dataset_preprocess/glm4/alpaca_converter.py \
   --data_path /path/alpaca_data.json \
   --output_path /path/alpaca_glm4_data.jsonl

  # 参数说明
  data_path:   输入下载的文件路径
  output_path: 输出文件的保存路径
  ```

  执行`mindformers/tools/dataset_preprocess/glm4/glm4_preprocess.py`文件，进行数据预处理和Mindrecord数据生成。

  ```shell
  python mindformers/tools/dataset_preprocess/glm4/glm4_preprocess.py \
   --input_glob /path/alpaca_glm4_data.jsonl \
   --vocab_file /path/tokenizer.model \
   --seq_length 8192 \
   --output_file /path/alpaca-messages.mindrecord

  # 参数说明
  input_glob:   转换后的alpaca的文件路径
  vocab_file:   tokenizer.model文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)
后进行使用。

| 模型名称          | MindSpore权重 |                   HuggingFace权重                    |
|:--------------|:-----------:|:--------------------------------------------------:|
| GLM-4-9B-Chat |      /      | [Link](https://huggingface.co/THUDM/glm-4-9b-chat) |
| GLM-4-9B      |      /      |   [Link](https://huggingface.co/THUDM/glm-4-9b)    |

注：词表文件为对应权重文件目录下tokenizer.model文件

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --torch_ckpt_path TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME

# 参数说明
torch_ckpt_path:  下载HuggingFace权重的文件夹路径
mindspore_ckpt_path: 转换后的MindSpore权重文件保存路径
```

## 全参微调

MindFormers提供`GLM4-9b`单机多卡微调示例，过程中使用`alpaca`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

设置如下环境变量：

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
```

### 单机训练

以`GLM4-9b`单机8卡微调为例，使用配置文件`configs/glm4/finetune_glm4_9b.yaml`。

执行如下命令启动微调任务。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/glm4/finetune_glm4_9b.yaml \
 --load_checkpoint /path/GLM4_9b.ckpt \
 --auto_trans_ckpt True \
 --train_dataset /path/alpaca.mindrecord \
 --run_mode finetune" 8
```

## 推理

MindFormers提供`GLM-4-9B-Chat`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡多batch推理。
注意：需添加环境变量使能atb的PA算子：export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=PagedAttention;
                               export MS_LLM_SEQ_LENGTH_INDEX=1;
                               export MS_LLM_FORCE_RESIZE_KERNELS=PagedAttention

```shell
bash scripts/examples/glm4/run_glm4_predict.sh CONFIG_PATH CKPT_PATH TOKENIZER

# 参数说明
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
TOKENIZER:   模型tokenizer文件路径
```

运行如下命令进行推理：

```shell
bash scripts/examples/glm4/run_glm4_predict.sh \
 path/to/glm4/predict_glm4_9b_chat.yaml \
 path/to/glm4.ckpt

# 推理结果
# [gMASK] <sop> <|user|>
# 晚上睡不着应该怎么办 <|assistant|>
# 晚上睡不着觉可能会影响第二天的精神状态和工作效率。以下是一些建议，可以帮助改善睡眠质量：
#
# 1. **规律作息**：尽量每天同一时间上床睡觉和起床，包括周末。
#
# 2. **放松身心**：
#    - **深呼吸**：尝试深呼吸练习，帮助身体放松。
#    - **冥想**：通过冥想放松心情，减少焦虑。
#    - **热水澡**：睡前洗个热水澡有助于身体放松。
#
# 3. **避免刺激性饮料和食物**：睡前避免咖啡、茶、巧克力等含有咖啡因的食品和饮料。
#
# 4. **减少屏幕时间**：睡前减少使用手机、电脑等电子设备，因为屏幕发出的蓝光可能会干扰睡眠。
#
# 5. **舒适的环境**：确保卧室安静、黑暗和适宜的温度。
#
# 6. **适量运动**：白天进行适量的运动有助于晚上更好地入睡，但避免在睡前进行剧烈运动。
#
# 7. **避免白天打盹**：如果白天需要休息，尽量控制在30分钟以内。
#
# 8. **建立睡前仪式**：如阅读、听轻音乐等，帮助大脑逐渐进入睡眠状态。
#
# 9. **咨询专业人士**：如果上述方法都无效，建议咨询医生或睡眠专家。
#
# 10. **心理调适**：有时候，失眠可能与心理因素有关，如焦虑、抑郁等，这时需要寻求心理咨询。
#
# 请根据自己的实际情况尝试这些方法，并注意观察效果。如果失眠问题持续存在，建议及时就医。 <|user|>

# [gMASK] <sop> <|user|>
# 使用python编写快速排序代码 <|assistant|>
# 下面是一个使用Python编写的快速排序算法的实现。快速排序是一种分而治之的算法，它通过一个基准值将数组分为两个子数组，一个包含小于基准值的元素，另一个包含大于基准值的元素，然后递归地对这两个子数组进行排序。
#
# ```python
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     else:
#         pivot = arr[0]
#         less = [x for x in arr[1:] if x <= pivot]
#         greater = [x for x in arr[1:] if x > pivot]
#         return quick_sort(less) + [pivot] + quick_sort(greater)
#
# # 示例
# array = [3, 6, 8, 10, 1, 2, 1]
# sorted_array = quick_sort(array)
# print(sorted_array)
# ```
#
# 这段代码定义了一个`quick_sort`函数，它接受一个列表`arr`作为参数。如果列表的长度小于或等于1，则它已经是有序的，所以直接返回。否则，选择列表的第一个元素作为基准值`pivot`，然后创建两个新的列表`less`和`greater`，分别包含小于和大于基准值的元素。最后，递归地对`less`和`greater`进行快速排序，并将结果与基准值连接起来返回。
#
# 示例中的`array`是一个未排序的列表，调用`quick_sort(array)`后，会得到一个排序后的列表`sorted_array`。 <|user|>
#
# [gMASK] <sop> <|user|>
# 你好呀！ <|assistant|>
# 你好👋！很高兴见到你，有什么可以帮助你的吗？ <|user|>
```
