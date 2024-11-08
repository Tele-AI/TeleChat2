
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

1. 收集训练数据，并组织成指定格式（见 [dummy_data/sft/single_turn_example.jsonl](dummy_data/sft/single_turn_example.jsonl)）：

   ```Json
   {"conversation": [{"role": "user", "content": "请描述一下如何正确规划个人理财。"}, {"role": "bot", "content": "正确规划个人理财需要以下几个步骤：\n1.了解自己的财务状况。这包括收入、支出、资产和负债等信息。了解自己的财务状况可以帮助人们更好地制定财务计划。\n2.设定财务目标。需要考虑短期目标和长期目标，例如以年为单位设定的支出计划、购房、购车等的长期目标。\n3.制定预算计划。在了解自己的财务状况并设定财务目标后，需要制定一个预算计划。这可以帮助人们控制支出、节省开支并达到财务目标。\n4.理性投资和储蓄。人们可以投资于股票、基金、房产或其他投资渠道以实现财务目标。但在投资前需了解相关知识并进行风险评估。同时还应储蓄一定金额，以应对突发事件或为达成某些目标做准备。\n5.审时度势，合理调整。财务计划需要不断地审时度势，根据实际情况做出调整，以达到最终的财务目标。需要注意财务状况的变化、投资的收益和风险等因素。\n通过以上五个步骤，人们可以做到合理规划个人理财，掌握自己的财务命运，更好地实现自己的财务目标。"}]}
   ```
2. 指定训练数据集之间的配比，用于数据采样（见 [dummy_data/sft_data_config.json](dummy_data/sft_data_config.json)）：

   ```json
   {
        "./dummy_data/sft//multi_turn_example.jsonl": 1.6,
        "./dummy_data/sft/single_turn_example.jsonl": 2.15,
        "./dummy_data/sft/tools_example.jsonl": 1.0
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

请将 监督微调 的数据组成成如下 **jsonl** 格式，见（[dummy_data/sft/single_turn_example.jsonl](dummy_data/sft/single_turn_example.jsonl) 、 [dummy_data/sft/multi_turn_example.jsonl](dummy_data/sft/multi_turn_example.jsonl)） 、 [dummy_data/sft/tools_example.jsonl](dummy_data/sft/tools_example.jsonl)）：

```json
{"conversation": [{"role": "user", "content": "请描述一下如何正确规划个人理财。"}, {"role": "bot", "content": "正确规划个人理财需要以下几个步骤：\n1.了解自己的财务状况。这包括收入、支出、资产和负债等信息。了解自己的财务状况可以帮助人们更好地制定财务计划。\n2.设定财务目标。需要考虑短期目标和长期目标，例如以年为单位设定的支出计划、购房、购车等的长期目标。\n3.制定预算计划。在了解自己的财务状况并设定财务目标后，需要制定一个预算计划。这可以帮助人们控制支出、节省开支并达到财务目标。\n4.理性投资和储蓄。人们可以投资于股票、基金、房产或其他投资渠道以实现财务目标。但在投资前需了解相关知识并进行风险评估。同时还应储蓄一定金额，以应对突发事件或为达成某些目标做准备。\n5.审时度势，合理调整。财务计划需要不断地审时度势，根据实际情况做出调整，以达到最终的财务目标。需要注意财务状况的变化、投资的收益和风险等因素。\n通过以上五个步骤，人们可以做到合理规划个人理财，掌握自己的财务命运，更好地实现自己的财务目标。"}]}
{"conversation": [{"role": "user", "content": "你喜欢哪种类型的电影？"}, {"role": "bot", "content": "作为AI，我不能喜欢电影。 但是，根据人们的兴趣，有许多类型的电影包括恐怖、喜剧、爱情、动作、科幻等等。 具体喜欢哪种电影类型还要根据个人口味来决定。"}, {"role": "user", "content": "你有哪些好看的电影推荐吗？"}, {"role": "bot", "content": "以下是几部我认为很好看的电影推荐给你：\n\n1. 《阿甘正传》：一部感人至深的电影，讲述了一个智商低下的男孩在生活中的奋斗和成长。\n\n2. 《辛德勒的名单》：一部关于二战纳粹大屠杀的电影，讲述了工厂老板辛德勒如何通过雇佣犹太人来帮助他们躲避屠杀的故事。\n\n3. 《楚门的世界》：一部关于虚拟现实和真实生活的电影，讲述了一个人在接受了虚假生活的洗脑后如何追求真实的故事。\n\n4. 《肖申克的救赎》：一部经典的励志电影，讲述了银行家安迪·杜弗雷因被冤枉入狱，在狱中与朋友们一起奋斗逐步实现自由的故事。\n\n5. 《星际穿越》：一部科幻电影，讲述了一群宇航员穿越虫洞去寻找新家园的故事，影片通过很多物理概念和哲学思考，使人感悟深刻。\n\n希望这些电影能给你带来愉悦和启示。"}, {"role": "user", "content": "你觉得这些电影中哪一部最好看？"}, {"role": "bot", "content": "作为AI，我没有感受和喜好，所以不能说哪一部电影更好看。但是，这些电影都有着不同的魅力和吸引力，可以根据不同个人的口味来选择自己喜欢的电影。无论哪一部电影，只要它能让你感受到一些东西，就是一部好的电影。"}]}
{"conversation": [{"role": "user", "content": "你好，我需要帮助来计算我的抵押贷款支付。我有一笔20万美元的贷款，每年的利率为3.5%，贷款期限为30年。"}, {"role": "bot", "content": "", "tool_calls": "[{\"name\": \"calculate_mortgage\", \"arguments\": {\"loan_amount\": 200000, \"interest_rate\": 3.5, \"loan_term\": 30}}]"}, {"role": "tool", "content": "{\"monthly_payment\": 898.09}"}, {"role": "bot", "content": "根据您提供的信息，您的月度抵押贷款支付大约为898.09美元。"}, {"role": "user", "content": "太好了！谢谢你的帮助。"}, {"role": "bot", "content": "不客气！如果你还有其他问题，随时提问。"}], "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"calculate_mortgage\", \"description\": \"计算每月抵押贷款支付\", \"parameters\": {\"type\": \"object\", \"properties\": {\"loan_amount\": {\"type\": \"number\", \"description\": \"贷款金额\"}, \"interest_rate\": {\"type\": \"number\", \"description\": \"年利率\"}, \"loan_term\": {\"type\": \"integer\", \"description\": \"贷款期限（年）\"}}, \"required\": [\"loan_amount\", \"interest_rate\", \"loan_term\"]}}}]"}
```

每一行为一条数据。非工具调用微调数据仅包含 conversation 字段，由多轮对话组成的 List。而工具调用数据除 conversation 外，还包含 tools 字段。
***注意！在工具调用数据中，bot 回复中的 tool_calls、单条数据的 tools 字段，必须是 Json 格式的字符串！***
***注意！在工具调用数据中，bot 回复中的 tool_calls、单条数据的 tools 字段，必须是 Json 格式的字符串！***
***注意！在工具调用数据中，bot 回复中的 tool_calls、单条数据的 tools 字段，必须是 Json 格式的字符串！***

### 数据配比文件

为方便进行数据采样与配比，可通过编写数据采样配置文件，方便快捷的指定样本的采样比例，如（[dummy_data/sft_data_config.json](dummy_data/sft_data_config.json)）：

```json
{
    "./dummy_data/sft//multi_turn_example.jsonl": 1.6,
    "./dummy_data/sft/single_turn_example.jsonl": 2.15,
    "./dummy_data/sft/tools_example.jsonl": 1.0
}
```

例如，该文件指明，对于数据文件 dummy_data/sft//multi_turn_example.jsonl 中的数据，将会被随机采样 1.6 次，假设数据总共 1000 条，则采样后训练数据为 1000 * 1.6 = 1600 条。

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
