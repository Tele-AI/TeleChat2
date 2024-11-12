
# Deepspeed for Telechat

## 项目结构

train.py 是一个基于 transformers、deepspeed 编写的，快速上手 Telechat 预训练（或继续训练）和监督微调的脚本，支持全参、LoRA、QLoRA等方式训练，支持单机多卡、多机多卡训练。

```plaintext
deepspeed/
├── deepspeed_configs               # 存放 deepspeed 配置文件，用于配置 zero1 至 zero3 训练模型
├── dummy_data                      # 示例数据
│   ├── pretrain                    # 示例预训练数据集
│   ├── sft                         # 示例sft训练数据集
│   ├── pretrain_data_config.json   # 示例预训练数据配比文件
│   └── sft_data_config.json        # 示例sft数据配比文件
│
├── train_scripts                   # 启动训练脚本
├── hostfile                        # 多机多卡节点配置文件
├── merge_lora.py                   # 合并lora训练权重脚本
├── README.md
└── train.py                        # 主训练代码，含数据处理部分
```

## 使用记录

实验环境

| Linux                                                            | Python | Torch  | Transformers | DeepSpeed | Flash-attn  | GPU           | Model               |
| ---------------------------------------------------------------- | ------ | ------ | ------------ | --------- | ----------- | ------------- | ------------------- |
| Ubuntu 18.04.6 LTS<br />(GNU/Linus 4.18.0-338.el8.x86_6x x86_64) | 3.10.8 | 1.13.1 | 4.30.0       | 0.9.3     | 2.0.0.post1 | A100-SXM4-40G | Telechat-7B-chat-hf |

脚本运行结果：

<table align="center">
    <tr align="center" valign="center">
        <th colspan="2" ><p>8 * A100-SXM4-40G<br>max_seq_length 256</p></th>
        <th>dtype</th>
        <th>全参(full)</th>
        <th>LoRA</th>
        <th>QLoRA</th>
    </tr>
    <tr align="center" valign="center">
        <td rowspan="4"><p>单机多卡<br>多机多卡</p></td>
        <td rowspan="2">SFT</td>
        <td >fp16</td>
        <td ><p>zero-1❌<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>zero-1✅<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>zero-1✅<br>zero-2✅<br>zero-3❌</p></td>
    </tr>
    <tr align="center" valign="center">
        <td >bf16</td>
        <td ><p>zero-1❌<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>zero-1✅<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>zero-1✅<br>zero-2✅<br>zero-3❌</p></td>
    </tr>
    <tr align="center" valign="center">
        <td rowspan="2">Pre-train</td>
        <td >fp16</td>
        <td ><p>zero-1❌<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>/</p></td>
        <td ><p>/</p></td>
    </tr>
    <tr align="center" valign="center">
        <td >bf16</td>
        <td ><p>zero-1❌<br>zero-2✅<br>zero-3✅</p></td>
        <td ><p>/</p></td>
        <td ><p>/</p></td>
    </tr>
</table>

## 快速使用

在 train_scripts 下，选择对应的训练脚本，并指定好训练数据及部分参数配置，即可一键启动训练。

以单机多卡监督微调举例：

1. 收集训练数据，并组织成指定格式（如 [dummy_data/sft/demo_single.jsonl](dummy_data/sft/demo_single.jsonl)）：

   ```Json
   {"system": "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。\n", "dialog": [{"role": "user", "content": "Create a programming algorithm that identifies the maximum value in a list of numbers.\n[2, 5, 8, 10, 20, 15]"}, {"role": "bot", "content": "One possible algorithm to find the maximum value in a list of numbers, written in pseudocode, is:\n\n```\nlist = [2, 5, 8, 10, 20, 15]     // Initialize a list of numbers\nmax_value = list[0]              // Set the first element in the list as the maximum value\n\nfor number in list               // Iterate over the list of numbers\n    if number > max_value        // If the current number is greater than the maximum value\n        max_value = number       // Update the maximum value with the current number\n\nprint(\"Maximum value: \", max_value)  // After the iteration, print the maximum value\n```\n\n### Output:\nMaximum value: 20"}], "multiturn": false, "tool": false}
   ```

2. 指定训练数据集之间的配比，用于数据采样（见 [dummy_data/sft_data_config.json](dummy_data/sft_data_config.json)）：

   ```json
   {
        "./dummy_data/sft/demo_tool.jsonl": 1.01,
        "./dummy_data/sft/demo_role.jsonl": 1.24,
        "./dummy_data/sft/demo_math.jsonl": 1.78,
        "./dummy_data/sft/demo_single.jsonl": 1.42,
        "./dummy_data/sft/demo_multi.jsonl": 1.2,
        "./dummy_data/sft/demo_code.jsonl": 1.0
    }
   ```

3. 在 train_scripts 下找到 [sft_single_node.sh](deepspeed/train_scripts/sft_single_node.sh) 并修改对应参数：

   ```bash
   # 配置可用GPU
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

   # 模型文件路径、输出路径、DeepSpeed配置路径、训练数据路径
   MODEL_PATH=$1
   OUTPUT_DIR=./test_output
   DEEPSPEED=./deepspeed_configs/ds_z2_config.json
   DATA_CONFIG_FILE=./dummy_data/sft_data_config.json

   # 训练配置
   # TRAIN_MODE 可选 full \ lora \ qlora
   TASK_TYPE=sft
   TRAIN_MODE=full

   # 训练参数配置
   NUM_TRAIN_EPOCHS=1
   PER_DEVICE_TRAIN_BATCH_SIZE=1
   GRADIENT_ACCUMULATION_STEPS=4
   LEARNING_RATE=3e-5
   MAX_SEQ_LENGTH=4096
   WARMUP_RATIO=0.05
   LOGGING_STEPS=10
   SAVE_STEPS=100
   LR_SCHEDULER_TYPE=cosine
   GRADIENT_CHECKPOINTING=true
   SEED=42
   FP16=true
   BF16=false
   OPTIM=adamw_apex_fused
   ```
4. 启动训练脚本即可：

   ```bash
   bash sft_single_node.sh
   ```

   或者

   ```shell
   nohup bash sft_single_node > train.log &
   ```

## 数据示例

关于数据格式的说明。

**注：可在 train.py 中根据训练环境实际情况配置 NUM_CPU_CORES 的值，以加快数据处理速度。**

### 预训练 & 继续训练

请将 预训练 & 继续训练 的数据组织成如下 **jsonl** 格式，见（[dummy_data/pretrain/pretrain_example.jsonl](dummy_data/pretrain/pretrain_example.jsonl)）：

```json
{"text": "训练数据1"}
...
{"text": "训练数据n"}
```

### 监督微调

请将 监督微调 的数据组成成如下 **jsonl** 格式，样例数据见 dummy_data/sft 文件夹 ：

```json
{
	    "system": "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。\n",
    "dialog": [
        {"role": "user", "content": "Create a programming algorithm that identifies the maximum value in a list of numbers.\n[2, 5, 8, 10, 20, 15]"},
        {"role": "bot", "content": "One possible algorithm to find the maximum value in a list of numbers, written in pseudocode, is:\n\n```\nlist = [2, 5, 8, 10, 20, 15]     // Initialize a list of numbers\nmax_value = list[0]              // Set the first element in the list as the maximum value\n\nfor number in list               // Iterate over the list of numbers\n    if number > max_value        // If the current number is greater than the maximum value\n        max_value = number       // Update the maximum value with the current number\n\nprint(\"Maximum value: \", max_value)  // After the iteration, print the maximum value\n```\n\n### Output:\nMaximum value: 20"}
    ],
    "multiturn": false,
    "tool": false}
```

以下字段均为**必填**字段：
	  - system：对话时的 system prompt；
	  - dialog：对话数据，每一条为一个 dict，包含 role、content 两个字段，分别为角色和对应的回复；
	  - multiturn：是否为多轮对话数据；
	  - tool：是否为工具调用数据。

### 数据配比文件

为方便进行数据采样与配比，可通过编写数据采样配置文件，方便快捷的指定样本的采样比例，如（[dummy_data/sft_data_config.json](dummy_data/sft_data_config.json)）：

```json
{
    "./dummy_data/sft/demo_tool.jsonl": 1.01,
    "./dummy_data/sft/demo_role.jsonl": 1.24,
    "./dummy_data/sft/demo_math.jsonl": 1.78,
    "./dummy_data/sft/demo_single.jsonl": 1.42,
    "./dummy_data/sft/demo_multi.jsonl": 1.2,
    "./dummy_data/sft/demo_code.jsonl": 1.0
}
```

例如，该文件指明，对于数据文件 dummy_data/sft//demo_math.jsonl 中的数据，将会被随机采样 1.42 次，假设数据总共 1000 条，则采样后训练数据为 1000 * 1.42 = 1420 条。

**注意，在启动训练时，脚本会根据该文件，首先对数据进行采样、tokenization 处理，并在输出文件夹下，保存 data_cache 文件。**

## 配置说明

### 训练参数配置

```bash
# 配置可用GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定可用 GPU 编号
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES # 设置可用 GPU 环境变量

# 模型文件路径、输出路径、DeepSpeed配置路径、训练数据路径
MODEL_PATH=$1 # 模型文件路径
OUTPUT_DIR=./test_output # 训练输出路径，训练时会在该路径下生成处理好的数据缓存文件
DEEPSPEED=./deepspeed_configs/ds_z2_config.json # deepspeed 训练配置文件
DATA_CONFIG_FILE=./dummy_data/sft_data_config.json # 训练数据配比文件，键为数据集路径，值为数据集采样比例

# 训练配置
TASK_TYPE=sft # 任务类型，pretrain 代表预训练或继续训练，sft 代表监督微调
TRAIN_MODE=full # 训练模式，full 代表全参训练，lora、qlora 则代表高效低秩训练

# 训练参数配置
NUM_TRAIN_EPOCHS=1 # 训练的 epoch 数量
PER_DEVICE_TRAIN_BATCH_SIZE=1 # 单个 GPU 的训练 batch size
GRADIENT_ACCUMULATION_STEPS=4 # 训练多少个 step 做梯度累积回传
LEARNING_RATE=3e-5 # 训练的学习率，在 lora、qlora 训练时，可以适当增大
MAX_SEQ_LENGTH=4096 # 训练时模型的最大上下文长度
WARMUP_RATIO=0.05 # warmup 比例
LOGGING_STEPS=10 # 每多少步显示一次log
SAVE_STEPS=100 # 每多少步进行一次训练结果存储
LR_SCHEDULER_TYPE=cosine # 学习率衰减类型
GRADIENT_CHECKPOINTING=true # 是否打开梯度 checkpoint，低资源情况建议打开
SEED=42 # 随机种子
FP16=true # 是否启用 fp16 精度训练，与 BF16 不能同时为 True
BF16=false # 是否启用 bf16 精度训练，与 FP16 不能同时为 True
OPTIM=adamw_apex_fused # 使用何种优化器，常见为 adamw

```

更多参数配置信息，请见 [HuggingFace - TrainingArguments](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/trainer#transformers.TrainingArguments) 。

在配置多机多卡训练时，可能涉及到 ssh 的密钥免密登陆配置。

### DeepSpeed 参数配置

```json
{
    "zero_optimization": {
        "stage": 3, // 开启 zero stage 3
        "offload_optimizer": {
            "device": "cpu", // 开启优化器参数卸载，并将优化器参数缓存至cpu
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu", // 开启模型参数卸载，并将模型参数缓存至cpu
            "pin_memory": true
        },
        "fp16": {
            "enabled": "auto"
        },
        "bf16": {
            "enabled": "auto"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 20,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true
}
```

关于 zero-1、zero-2、zero-3 的配置原则，在保障资源的情况下，优先级请按照：zero-1 > zero-2 > zero-3

更多关于 DeepSpeed 的参数释义及配置，请见 [HuggingFace -DeepSpeed](https://huggingface.co/docs/transformers/main/deepspeed) 。
