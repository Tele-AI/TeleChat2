# 文本生成推理

Mindformers大模型套件提供了text generator方法，旨在让用户能够便捷地使用生成类语言模型进行文本生成任务，包括但不限于解答问题、填充不完整文本或翻译源语言到目标语言等。

当前该方法支持Minformers大模型套件中6个生成类语言模型

## [Text Generator支持度表](../model_support_list.md#text-generator支持度表)

## 增量推理

Mindformers大模型套件的`text generator`方法支持增量推理逻辑，该逻辑旨在加快用户在调用`text generator`方法进行文本生成时的文本生成速度。

在此提供使用高阶接口进行各模型增量推理的**测试样例脚本**：

```python
# mindspore设置图模式和环境
import mindspore; mindspore.set_context(mode=0, device_id=0, jit_config={"infer_boost": "on"})
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 按需设置模型类型名，高阶接口将根据类型名实例化相应模型
model_type = "glm_6b"
# 按需设置测试的输入文本
input_text = "中国的首都是哪个城市？"

# 获取模型默认配置项并按需修改
config = AutoConfig.from_pretrained(model_type)
# use_past设置为True时为增量推理，反之为自回归推理
config.use_past = True
# 修改batch_size和模型seq_length
config.batch_size = 1; config.seq_length=512

# 根据配置项实例化模型
model = AutoModel.from_config(config)
# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type)
# 对输入进行tokenizer编码
input_ids = tokenizer(input_text)["input_ids"]
# 调用model.generate接口执行增量推理
output = model.generate(input_ids, max_length=128, do_sample=False)
# 解码并打印输出
print(tokenizer.decode(output))
```

> 注：
>
> 1. 首次调用generate时需要进行mindspore图编译，耗时较长；在统计在线推理的文本生成速度时，可以多次重复调用并排除首次调用的执行时间
> 2. 使用增量推理(use_past=True)时的生成速度预期快于自回归推理(use_past=False)

## Batch推理

`text generator`方法也支持同时对多个输入样本进行batch推理；在单batch推理算力不足的情况下，多batch推理能够提升推理时的吞吐率

以下给出测试batch推理能力的**标准测试脚本**，仅上述增量推理测试脚本仅有少数区别

```python
import mindspore;mindspore.set_context(mode=0, device_id=0, jit_config={"infer_boost": "on"})
from mindformers import AutoConfig, AutoModel, AutoTokenizer

model_type = "glm_6b"
# 多batch输入文本
input_text = [
    "中国的首都是哪个城市？",
    "你好",
    "请介绍一下华为",
    "I love Beijing, because"
]
# 是否使用增量推理
use_past = True
# 预设模型seq_length
seq_len = 512

config = AutoConfig.from_pretrained(model_type)
# 将batch size修改为输入的样本数
config.batch_size = len(input_text)
config.use_past = use_past
config.seq_length = seq_len

model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_type)

# 对输入batch进行pad
input_ids = tokenizer(input_text, max_length=config.seq_length, padding="max_length")["input_ids"]
output = model.generate(input_ids, max_length=128, do_sample=False)
print(tokenizer.decode(output))
```

> 注：
> batch推理的推理吞吐率提升表现与设备计算负荷相关；在seq_len较短并开启增量推理的情况下，计算负荷较小，使用batch推理通常会获得较好的提升

## 流式推理

Mindformers大模型套件提供Streamer类，旨在用户在调用text generator方法进行文本生成时能够实时看到生成的每一个词，而不必等待所有结果均生成结束。

实例化streamer并向text generator方法传入该实例：

```python
from mindformers import AutoModel, AutoTokenizer, TextStreamer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

streamer = TextStreamer(tok)

_ = model.generate(inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
# 'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```

上述脚本不再对输出文本进行统一解码打印，而是每生成一个中间结果就由streamer实时打印

## 分布式推理

对于无法在单卡上完成部署的大模型，需要通过多卡分布式，对模型进行切分后再进行推理

当前分布式推理相较于单卡推理，流程明显更为复杂，不易使用

这里介绍**文本生成任务分布式推理**的指导流程，以期对各模型的分布式推理使用起到指导与参考作用

### 分布式推理概述

分布式推理与单卡推理有以下几个区别点：

1. 分布式推理时，需要多卡多进程拉起同一推理任务
2. 分布式推理时，模型在进行推理前需要按设定的分布式策略进行切分
3. 分布式推理时，切分的模型加载的权重也需要为切分权重

由于上述几点区别点的存在，分布式推理需要在单卡推理的基础上进行更多准备工作，如准备分布式启动脚本，推理前对模型权重进行切分，推理代码需修改以适配模型切分与加载分布式权重等；流程相对繁杂，下文将逐一介绍。

### 前期准备

#### 生成RANK_TABLE_FILE

分布式推理需要分布式多卡启动进程，为此需提前准备RANK_TABLE_FILE文件

运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件，其中 `[0,8)` 可以替换为实际使用的卡区间

```bash
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

注：若使用ModelArts的notebook环境，可从 /user/config/jobstart_hccl.json 路径下直接获取rank table，无需手动生成

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

#### 确认分布式切分策略

分布式推理通常用于在单卡上无法完成部署的大模型，使用多卡切分进行推理部署

使用分布式推理时，需明确所使用的切分策略，通常是dp=1，mp=卡数，(pp分布式推理尚不支持)

**对于数据并行dp**，表示多卡推理时，不同dp间使用了不同的数据执行推理，其效果等价于在不同卡上各自运行推理任务，而仅输入不同；
因此dp切分仅适用于同时有大量数据需要获取推理结果的场景；对于单卡能够完成推理的模型，建议使用增大batch的方式来增加吞吐量，以代替dp分布式推理；
注意，在dp场景下，generate输入的文本bs应当被dp整除，如dp=2，则输入的文本总条数应至少为2条，且模型bs设置与文本bs一致

**对于模型并行mp**，表示多卡推理时，将模型权重切分为mp份放在不同的卡上进行计算，相当于多卡同时处理一项推理任务；因此mp切分更适用于单卡无法部署的大模型推理

明确分布式推理所需的卡数与分布式策略，关系为 `dp * mp = 卡数`；该切分策略将在后续流程中使用到。

#### 模型权重切分

在分布式推理场景下，常需要将模型权重重新切分以适应目标切分策略，常见场景为：

**场景一**：从完整模型权重切分至分布式权重

通常是已有完整权重，但目标切分策略存在mp切分，需要将权重切分为对应mp策略份，此时可参考[权重转换文档](./Transform_Ckpt.md)，生成目标strategy，将完整权重转换为目标切分权重

**场景二**：从分布式训练获得的已切分权重转化为另一策略的分布式权重

通常是在分布式训练完成后获取了按训练切分策略进行切分的权重，在推理阶段模型需要转换为另一切分策略；
同样可参考[权重转换文档](./Transform_Ckpt.md)，生成目标strategy，与原有切分startegy一同，转换模型切分策略

### 分布式推理脚本

#### 基于generate接口的自定义推理脚本

我们在 `scripts/examples/distribute_generate` 文件夹下提供了基于generate接口的自定义推理脚本 `generate_custom.py`，支持分布式推理。

在此对脚本中适配分布式的几个要点进行讲解，方便用户理解分布式推理流程，并能够根据实际需求改写脚本，实现自定义分布式推理流程。

1. 分布式context环境初始化

    ```python
    def context_init(use_parallel=False, device_id=0):
        """init context for mindspore."""
        context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
        parallel_config = None
        if use_parallel:
            parallel_config = ParallelContextConfig(
                parallel_mode='SEMI_AUTO_PARALLEL',     # 默认使用半自动并行模式
                gradients_mean=False,                   # 推理不涉及梯度平均
                full_batch=True                         # 半自动并行默认开启full batch
            )
        # 初始化context环境
        rank_id, device_num = init_context(
            use_parallel=use_parallel,
            context_config=context_config,
            parallel_config=parallel_config
        )
        print(f"Context inited for rank {rank_id}; total device num is {device_num}.")
    ```

    调用init_context接口，在分布式场景下正确设置半自动并行场景

2. 模型配置分布式策略

    ```python
    # 2.3 配置模型切分策略，当前暂不支持pipeline并行策略
    parallel_config = TransformerOpParallelConfig(
        data_parallel=args.data_parallel,
        model_parallel=args.model_parallel
    )
    model_config.parallel_config = parallel_config
    ```

    根据确定好的推理阶段的分布式策略，对模型配置相应的分布式策略

3. 分布式加载切分权重

    ```python
    # 2.4 分布式推理时需通过分布式接口加载权重，移除原权重路径以避免在模型实例化时加载
    if args.use_parallel:
        model_config.checkpoint_name_or_path = None
    ...
    # 4. 分布式下，模型编译切分并加载权重
    # if use parallel, load distributed checkpoints
    if args.use_parallel:
        print("---------------Load Sharding Checkpoints---------------", flush=True)
        load_sharding_checkpoint(args.checkpoint_path, network, model_config)
    ...
    def load_sharding_checkpoint(checkpoint_path, network, model_config):
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"checkpoint_path {checkpoint_path} is not a directory, which is required for distribute "
                            "generate, please check your input checkpoint path.")
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print(f"ckpt path: {str(ckpt_path)}", flush=True)

        # shard model and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print(f"Network parameters are not loaded: {str(not_load_network_params)}", flush=True)
    ```

    分布式推理场景下，模型在加载权重前，需调用 `infer_predict_layout` 接口，进行模型编译与分布式切分；模型切分后，再按rank_id加载各自的切分权重，这样就完成了分布式的权重加载，解决了大模型无法单卡完成加载的问题

以上3点是分布式推理脚本流程中与单卡推理的主要区别点，其余流程则与单卡推理基本一致，实例化模型和相应tokenizer后，调用 `.generate()` 接口完成推理流程

#### 低参微调模型修改点

如为低参微调模型，仅需在实例化模型前，配置模型对应的低参微调参数，实例化出对应的低参模型即可；

样例脚本如下：

```python
# 以llama_lora为例，其pet config配置如下
pet_config = {
    "pet_type": "lora"
    # configuration of lora
    "lora_rank": 16
    "lora_alpha": 16
    "lora_dropout": 0.05
    "target_modules": '.*wq|.*wk|.*wv|.*wo'
}
model_config.pet_config = pet_config
# 设置模型配置的pet_config项后，from_config接口将会实例化出带lora结构的模型
network = AutoModel.from_config(model_config)
```

#### 分布式启动shell脚本

我们提供了启动分布式推理的参考shell脚本 `scripts/examples/distribute_generate/run_dist_gen.sh`

脚本启动命令为：

```bash
bash run_dist_gen.sh [EXECUTE_ORDER] [RANK_TABLE_PATH] [DEVICE_RANGE] [RANK_SIZE]
```

各项入参含义为：

1. `EXECUTE_ORDER`：需执行的命令，可以字符串形式传入完整的python命令
2. `RANK_TABLE_PATH`：rank table文件的路径
3. `DEVICE_RANGE`：期望使用的卡号范围，如 `[0,8]` 表示使用编号为0到7的共8张卡
4. `RANK_SIZE`：使用的总卡数

#### 执行分布式推理

**样例1**：

以gpt2模型为例，拉起两卡模型并行分布式推理：

```bash
export INPUT_DATA="An increasing sequence: one,"
bash run_dist_gen.sh "python generate_custom.py --model_type gpt2 --checkpoint_path ./gpt2_ckpt --use_parallel True --data_parallel 1 --model_parallel 2" /path/to/hccl_2p_xxx.json '[0,2]' 2
```

参数含义:

- `export INPUT_DATA="An increasing sequence: one,"`：使用环境变量的方式输入文本，shell脚本将该项作为python脚本的input_data输入；主要目的是规避字符串输入与shell解析存在的问题，目前需要用户手动控制

- `python generate_custom.py --model_type gpt2 --checkpoint_path ./gpt2_ckpt --use_parallel True --data_parallel 1 --model_parallel 2"`: 需执行的命令，此处完整输入generate_custom.py的启动命令

python generate_custom.py 各项参数含义：

- `model_type`: 使用的模型类型，此处选择 `gpt2` 模型
- `checkpoint_path`: 权重路径，此处替换为实际需加载的权重路径，注意，需按照[权重切分](#模型权重切分)进行切分，并输入权重文件夹路径，路径下目录结构组织类似 `./gpt2_ckpt/rank_xx/xxx.ckpt`
- `use_parallel`: 是否使用多卡并行推理，此处为 `True`
- `data_parallel`: 数据并行数，此处为1表示不开启
- `model_parallel`: 模型并行数，此处为2表示2卡并行
- `input_data`: 推理输入的文本

bash 脚本其余参数：

- `/path/to/hccl_2p_xxx.json`: rank table file路径，替换为之前准备的rank table file的实际路径
- `'[0,2]'`: 占用的卡范围，0包含，2不包含，表示使用 `0~1` 2张卡并行推理
- `2`: rank size，一共使用了多少张卡，此处为2

输出日志：

```text
['An increasing sequence: one, two, three, four, five. And so on.\n\nThe first is the first sequence of the second sequence, which is called the first and second sequence.\n\nThe second sequence is called the third and fourth sequence, and so on.\n\nThe third and fourth sequence is called the first and second sequence, and so on. The fourth sequence is called the first and second sequence, and so on.\n\nThe fifth sequence is called the second and third sequence, and so on.\n\nThe sixth sequence is called the third and fourth sequence, and so on.\n\nThe seventh sequence is called the second']
```

**样例2**：

gpt2模型测试dp推理

`input.txt` 文本内容：

```text
An increasing sequence: one,
I love Beijing,
```

拉起脚本：

```bash
export INPUT_DATA=input.txt
bash run_dist_gen.sh "python generate_custom.py --model_type gpt2 --batch_size 2 --checkpoint_path ./gpt2_ckpt --use_parallel True --data_parallel 2 --model_parallel 1" /path/to/hccl_2p_xxx.json '[0,2]' 2
```

与mp切分区别点：

1. 输入文本条数需为dp倍数，此处dp为2，因此准备两条输入
2. `batch_size`设置为2，与输入文本输入条数匹配
3. 纯dp权重与mp切分不一致，可以将原权重按预期文件结构组织 `./gpt2_ckpt/rank_xx/xxx.ckpt`
4. 修改模型并行策略，dp=2，mp=1

输出日志：

```text
["An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.\n\nThe first is the most important. The first is the most important because it's the one that's most important. The first is the most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one that's most important because it's the one", 'I love Beijing, but I\'m also a bit worried that it might become the next Hong Kong," said Mr. Wang.\n\n"I think the government is going to be very worried, but I don\'t know what will happen. We\'re going to be in a situation where we\'re going to have to deal with the Chinese people, and I think that\'s what we\'re going through," he added.\n\nChina has been a major player in international trade since the late 1980s, when it was the world\'s second-largest economy. But in recent years the Chinese have become more assertive in their economic and political relations with other nations, including the']
```

## SLoRA推理

LoRA（Low-Rank Adaptation）是一种轻量化的参数高效微调技术，它通过将权重矩阵分解为两个较低秩的矩阵，显著减少模型微调时所需的参数量。这种方法特别适用于大型语言模型，因为LLM通常拥有数以亿计甚至数以十亿计的参数，传统微调所有参数的方式不但耗时，且需要大量存储和计算资源。而LoRA的出现，解决了这一问题，使得在保持模型性能的同时，降低了计算开销。

在现实场景中，LLM往往需要同时应对不同的任务。例如，在对话生成、机器翻译、代码生成等任务中，虽然可以使用相同的基准LLM模型，但每个任务可能需要不同的微调策略。通过使用多LoRA，可以为每个任务加载不同的低秩权重调整，从而在保持基准模型不变的基础上，高效处理多个任务。这不仅节省了存储空间，还减少了训练和推理时的计算负担。

这里介绍**文本生成任务SLoRA推理**的指导流程，以期对模型的SLoRA推理使用起到指导与参考作用

### SLoRA场景分析

1. 推理过程中，用户需要使用同一个预训练的基准模型与多个不同的Adapter权重
2. 用户具备多种Adapter微调权重，且微调以LoRA的方式进行
3. 针对不同的请求，需要使用共享参数的相同的基础模型，但不同的微调权重进行推理

### 前期准备

#### Adapter文件配置

需要为每个Adapter微调权重配置不同的config.json文件:

```json
{
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "r": 8,
  "target_modules": [
    "wq",
    "wv",
    "wk",
    "wo",
    "w1",
    "w2",
    "w3"
  ]
}
```

|    参数    |    参数说明        |
| :--------: | :-------------------: |
| lora_alpha | 指定LoRA的缩放因子 |
| lora_dropout | 指定LoRA的dropout比例 |
| r | 指定低秩矩阵的秩 |
| target_modules | 表示LorA将应用的特定的模型模块 |

注意，微调权重需要为满足MindFormers网络结构的ckpt

不同Adapter微调权重需要分别存放在不同目录下，并且需要搭配以对应的config.json文件，且目录下Adapter文件名一致为adapter_model.ckpt

```text
目录层次：
└─Adapter1
    ├─adapter_model.ckpt
    └─adapter_config.json
 ```

#### lora_adapter.json文件配置

按照以下格式依据各Adapter目录路径配置lora_adapter.json。其中，key为Adapter的名称，value为LoRA权重的路径，用于推理时选择对应的LoRA权重。

```json
{"adapter1": "/path/to/adapter1_directory",
"adapter2": "/path/to/adapter2_directory"}
```

#### 推理yaml文件配置

我们在Pet微调模型框架中增加了SLoRA推理模型，可以通过在配置文件的模型参数中增加以下字段实现SLoRA推理：

```yaml
model:
    xxx:
    model_config:
        pet_config:
            pet_type: slora
            adapter_path: "/path/to/your/lora_adapter.json"
    xxx:
```

### 基于单机单卡的推理脚本

```bash
    python run_mindformer.py \
    --use_parallel False \
    --run_mode predict \
    --adapter_id 'Adapter1' \
    --config configs/llama2/predict_llama2_7b.yaml \
    --predict_data 'I love Beijing because'
```

其中，`adapter_id`指定用于推理的LoRA Adapter名称，若为None或lora_adapter.json中不存在的名称则使用基础模型

### 基于单机多卡的推理脚本

```bash
    bash scripts/msrun_launcher.sh "run_mindformer.py \
    --use_parallel True \
    --run_mode predict \
    --adapter_id 'Adapter1' \
    --config configs/llama2/predict_llama2_7b.yaml \
    --predict_data 'I love Beijing because' " 8
```

当使用多卡并行推理时，需要将use_parallel设置为True，打开分布式加载切分lora权重，具体场景为：

1、auto_trans_ckpt字段为True时，将自动切分lora权重，切分后的分布式权重保存在output/transform_ckpt/slora

2、auto_trans_ckpt字段为False时，需要将分布式lora权重保存在lora_adapter.json中的第一个adapter路径下。
