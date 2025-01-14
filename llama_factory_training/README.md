# LLaMA-Factory
我们将介绍如何使用 LLaMA-Factory 微调 TeleChat2 模型。

* 支持单卡和多卡分布式训练
* 支持全参数微调、LoRA、Q-LoRA 和 DoRA 。

# 安装
开始之前，确保你已经安装了以下代码库：

1. 根据 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 官方指引构建好你的环境
2. 安装下列代码库（可选）：

    ```bash
    pip install deepspeed
    pip install flash-attn --no-build-isolation
    ```

3. 如你使用 [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)  ，请确保你的CUDA版本在11.6以上。

# 准备数据
LLaMA-Factory 在 data 文件夹中提供了多个训练数据集，您可以直接使用它们。如果您打算使用自定义数据集，请按照以下方式准备您的数据集。

1. 请将您的数据以 json 格式进行组织，并将数据放入 data 文件夹中。LLaMA-Factory 支持以 alpaca 或 sharegpt 格式的数据集。

* alpaca 格式的数据集应遵循以下格式：
    ```json
    [
        {
            "instruction": "user instruction (required)",
            "input": "user input (optional)",
            "output": "model response (required)",
            "system": "system prompt (optional)",
            "history": [
                ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                ["user instruction in the second round (optional)", "model response in the second round (optional)"]
            ]
        }
    ]
    ```
* sharegpt 格式的数据集应遵循以下格式：
    ```json
    [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "user instruction"
                },
                {
                    "from": "gpt",
                    "value": "model response"
                }
            ],
            "system": "system prompt (optional)",
            "tools": "tool description (optional)"
        }
    ]
    ```
2. 在 data/dataset_info.json 文件中提供您的数据集定义，并采用以下格式：

* 对于 alpaca 格式的数据集，其 dataset_info.json 文件中的列应为：

    ```json
    "dataset_name": {
    "file_name": "dataset_name.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
        }
    }
    ```
* 对于 sharegpt 格式的数据集，dataset_info.json 文件中的列应该包括：

    ```json
    "dataset_name": {
        "file_name": "dataset_name.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
    ```

# 训练
下载模型，例如模型位置为："./telechat_7B"，需要将模型位置中的"modeling_telechat2.py"文件进行替换，使用[modeling_telechat2.py](../llama_factory_training/modeling_telechat2.py)文件替换即可。

替换完毕后，使用如下命令即可使用示例数据进行lora微调过程。

```Bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \--stage sft \--do_train True \--model_name_or_path ./telechat_7B \--preprocessing_num_workers 16 \--finetuning_type lora \--template telechat \--flash_attn auto \--dataset_dir data \--dataset identity,alpaca_en_demo \--cutoff_len 1024 \--learning_rate 3.e-5 \--num_train_epochs 2.0 \--max_samples 100000 \--per_device_train_batch_size 1 \--gradient_accumulation_steps 1 \--lr_scheduler_type cosine \--max_grad_norm 1.0 \--logging_steps 1 \--save_steps 100 \--warmup_steps 0 \--optim adamw_torch \--packing False \--report_to none \--output_dir saves/telechat_7B_lora \--plot_loss True \--ddp_timeout 180000000 \--include_num_input_tokens_seen True \--lora_rank 8 \--lora_alpha 16 \--lora_dropout 0 \--lora_target all 
```

该命令指定学习率为3e-5，训练epoch为2，训练后保存的权重位置为**"saves/telechat_7B_lora"**。

## 对话

微调完后，即可开始对话，首先需要定义配置文件：**"inference_telechat_lora_sft.yaml"**

```Bash
model_name_or_path: ./telechat_7B
adapter_name_or_path: saves/telechat_7B_lora
template: telechat
infer_backend: huggingface  # choices: [huggingface, vllm]
```

使用该命令启动推理脚本，该方式会自动合并lora部分权重：

```Bash
llamafactory-cli chat inference_telechat_lora_sft.yaml
```
运行后，即可在命令行中进行输入，并得到推理结果。

## lora权重合并

如果想导出新模型进行后续推理与部署，需要进行lora权重合并，首先定义配置文件：**"merge_telechat_lora_sft.yaml"**

```Bash
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters
### model
model_name_or_path: ./telechat_7B
adapter_name_or_path: saves/telechat_7B_lora
template: telechat
finetuning_type: lora

### export
export_dir: models/telechat_7B_lora
export_size: 2
export_device: cpu
export_legacy_format: false
```

使用该命令运行合并脚本：

```Bash
llamafactory-cli export merge_telechat_lora_sft.yaml
```

运行该脚本后，则会在当前目录下生成: **"models/telechat_7B_lora"**目录，该目录为导出的新模型，里面会包含配置文件，字典文件，以及权重文件等，可用于后续的推理与部署。

## huggingface 推理

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained('models/telechat_7B_lora', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('models/telechat_7B_lora', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)

prompt = "生抽与老抽的区别？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response
```

## vllm部署

合并权重后，新模型位于: **"models/telechat_7B_lora"**

```Bash
vllm serve models/telechat_7B_lora --trust-remote-code
```

通过该命令即可部署成功。