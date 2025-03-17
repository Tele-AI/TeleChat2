# MoE模型

### MoE模型介绍

- MoE模型**TeleChat2-39B-A12B**是由中国电信人工智能研究院研发训练的大语言模型，该系列模型**完全基于国产算力**训练。

- 训练数据方面，**TeleChat2-39B-A12B**模型采用5万亿 Tokens中英文高质量语料进行训练。

- 框架优化方面，在MOE模块将Tensor并行域转换成专家并行域，从而将MOE的AllToAll 通讯约束在节点内，提高通讯效率; 把MOE输入切成多个副本依次下发，将dispatch通信/FFN计算/combine通信三个环节连成流水线，实现moe的计算通信掩盖; 基于对内存和计算的开销建模，理论求解内存约束下性能最优的流水线并行的负载配置，实现流水线负载均衡。

- 训练方面，采用课程学习的方式，首先聚焦低难度、高质量教育知识以及多语言数据进行模型训练，以获得较好的模型初始性能；然后引入复杂数据，增大数学、逻辑推理、代码等数据占比，提升模型逻辑推理能力；最后，使用高质量数据进行退火，持续提升模型效果；

### 模型结构

| layer_num | num_experts | num_chosen_experts | hidden_size | ffn_hidden_size | head_num | tie_word_embeddings | GQA  |
| --------- | ----------- | ------------------ | ----------- | --------------- | -------- | ------------------- | ---- |
| 30        | 16          | 4                  | 4096        | 6144            | 32       | 否                  | 否   |

### 模型地址

本次同时发布GPU与NPU版本模型，下载链接见下表：

| 模型版本               | 下载链接                                                     |
| ---------------------- | ------------------------------------------------------------ |
| TeleChat2-39B-A12B-GPU | [modelscope](https://modelscope.cn/models/TeleAI/TeleChat2-39B-A12B) |
| TeleChat2-39B-A12B-NPU | [modelscope](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_7B/Telechat_39B_A12.zip) |


### 效果评测

综合评测数据集上，TeleChat2-39B-A12B模型以12B激活参数量接近TeleChat2-35B模型效果。

| Dataset    | TeleChat2-35B | TeleChat2-39B-A12B | TeleChat2-7B | TeleChat2-3B |
| ---------- | ------------- | ------------------ | ------------ | ------------ |
| C-Eval     | 85            | 89                 | 82           | 75           |
| MMLU       | 82            | 83                 | 79.6         | 72.9         |
| CMMLU      | 90.18         | 90                 | 84.6         | 73           |
| GSM8K      | 91            | 83.5               | 86.8         | 64.7         |
| HumanEval  | 73            | 68                 | 56           | 38           |
| MBPP       | 75            | 67                 | 62.6         | 47           |
| AlignBench | 7.88          | 7.56               | 6.96         | 5.74         |
| IFEval     | 79.63         | 76.48              | 73.1         | 61.29        |

### 模型推理



```python
import os 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PATH = "./TeleChat2-39B-A12B" # model download path
tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float32)

prompt = "高音单簧管跟高音萨克斯的调性相同吗？如果相同，请说出他们的调性，如果不同，请分别说出他们的调性。"
messages = [{"role": user, "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, 
                              max_new_tokens=4096, 
                              repetition_penalty=1.0,
                              do_sample=Falese)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(respons)

```



### 国产化适配

#### 昇腾Atlas 800T A2训练服务器实现训练、推理适配

##### 核心组件：

- 昇思MindSpore：该框架是华为开发的深度学习框架，旨在为AI应用提供高效、灵活的开发环境。它支持多种硬件平台，并具有自动微分、模型优化等功能，适合各种深度学习任务。

- MindSpore Transformers：该框架的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

**当前星辰语义大模型TeleChat2支持昇腾Atlas 800T A2训练服务器，可基于昇思MindSpore框架以及MindSpore Transformers框架进行模型训练和评测，详情请看[telechat国产化](../tutorial/telechat_国产化运行.md)。如果您对mindsformers相关特性有疑问，也可以查看[mindformers](https://gitee.com/mindspore/mindformers/tree/dev/)。**
