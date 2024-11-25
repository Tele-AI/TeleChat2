# 长序列训练

从生成性AI到科研模型，长序列训练正在变得非常重要。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度（S）增长时，训练内存开销会以O（$S^2$）的速度增长。因此需要针对长序列场景进行特定的优化解决长训练场景的训练需求。mindformers提供了一种显存高效的序列并行方法，用于减少输入序列长度的限制。同时，支持了attention_mask压缩特性，解决长序列场景下生成超大attention_mask矩阵的显存占用问题，上述特性能够有效地支持超长序列训练。

## 使用说明

当前特性支持的模型如下所示：
|    模型    |          参数量           |
| :--------: | :-----------------------: |
|   LLaMA2   | 7B<br>13B<br>70B |
|   Qwen     | 7B<br>14B<br>72B |
|   Qwen1_5  | 7B<br>14B<br>72B |

注：当前长序列并行切分能力仅支持与Flash Attention特性共同使用，请确保模型已经支持FA并开启了`use_flash_attention`配置

## 配置序列并行

并行方法在序列维度进行切分，每台设备只负责1/CP的Q和KV进行自注意力值计算，不再需要单个设备来保存整个序列。注意力矩阵与序列长度由平方关系，变成线性关系。有效降低每台计算设备显存压力。同时，该方法与数据并行、流水线并行和模型并行兼容，但需要满足实际运行的卡数 device_num = data_parallel × model_parallel × context_parallel × pipeline_stage，context_parallel代表序列并行数，此处为1表示不开启，此处为2表示2卡并行。另外，需要注意本序列并行方案是为了减少长序列对单设备的限制，而use_seq_parallel使能的序列并行方案是为了处理张量并行中未能分摊的显存，将Transformer层中的LayerNorm以及Dropout的输入按序列维度进行切分，使各设备只需处理部分的LayerNorm和Dropout，实现用更少的设备训练大模型，两种序列并行方案暂不兼容，因此使能本序列并行方案时需要在配置文件中将parallel_config.use_seq_parallel设为False。

- **data_parallel**: 数据并行
- **model_parallel**: 模型并行
- **context_parallel**: 序列并行
- **pipeline_stage**: 流水线并行

序列并行配置参考样例：

```yaml
parallel_config:
  context_parallel: 2
  use_seq_parallel: False
```

## 配置attention_mask压缩

attention_mask压缩是对Self-Attention中的Score矩阵进行掩码操作,它的内存大小跟 呈正比。例如在32k序列下，单个uint8类型的attention_mask矩阵会占用1GB的显存，使能后传入的attention_mask为优化后的压缩下三角矩阵（2048*2048）。除内存收益外，有些网络会在device上生成attention_mask矩阵，attention_mask压缩能够有效地避免生成超大矩阵带来的性能开销。当使用attention_mask压缩时需要在配置文件中将model.model_config.use_attn_mask_compression设为True。

attention_mask压缩配置参考样例：

```yaml
# model config
model:
  model_config:
    use_attn_mask_compression: True
```

## Ulysses序列并行方案

DeepSpeed提出的[Ulysses长序列并行方案](https://arxiv.org/abs/2309.14509)，将各个样本在seq维度切分给不同的计算卡；然后，在attention计算之前，对QKV执行all-to-all通信操作，以使每个计算卡接收完整的序列，使得各计算卡可以并行计算不同的注意力头；最后，在attention计算后使用另一个all-to-all来在注意力头上收集结果，同时重新在seq维度上进行切分。

该方案可以有效扩展训练的序列长度，同时保持相对较低的通信量

MindFormers已支持配置Ulysses序列并行方案，可通过以下配置项使能Ulysses序列并行：

```yaml
parallel:
  ...
  enable_alltoall: True  # 允许插入alltoall算子
  ...
parallel_config:
  ...
  context_parallel: 2
  context_parallel_algo: ulysses_cp  # 使能ulysses序列并行
  ...
```

其中，`context_parallel_algo` 参数为枚举类，表示选定的序列并行算法；目前支持 `colossalai_cp` 和 `ulysses_cp` 两种配置，分别表示使用colossal-ai的序列并行方案和ulysses序列并行方案；
默认值为 `colossalai_cp`；因此要使用ulysses方案时，需将该项配置为 `ulysses_cp`。

`enable_alltoall`项表示允许生成alltoall通信算子，不启用时将会由allgather等其他算子组合完成等价替代，可参考MindSpore `set_auto_parallel_context`[接口文档](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.set_auto_parallel_context.html)；启用ulysses方案时我们期望能够直接插入alltoall通信算子，因此将该配置项打开。

注意：ulysses方案需要在`context_parallel`切分数大于1的场景下使用
