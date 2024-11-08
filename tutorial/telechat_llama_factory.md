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

需要将模型位置中的"modeling_telechat2.py"文件进行替换，使用[modeling_telechat2.py](../llama_factory_training/modeling_telechat2.py)文件替换即可。

执行下列命令：

```bash
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path $MODEL_PATH \
    --dataset your_dataset \
    --template telechat \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3
```
并享受训练过程。若要调整您的训练，您可以通过修改训练命令中的参数来调整超参数。其中一个需要注意的参数是 cutoff_len ，它代表训练数据的最大长度。通过控制这个参数，可以避免出现OOM（内存溢出）错误。

合并LoRA
如果你使用 LoRA 训练模型，可能需要将adapter参数合并到主分支中。请运行以下命令以执行 LoRA adapter 的合并操作。

```python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path path_to_base_model \
    --adapter_name_or_path path_to_adapter \
    --template telechat \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```