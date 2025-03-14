# 流水并行的负载偏置与重计算

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

在大模型训练调优过程中，设备内存的合理使用和分配是一项重要的环节，除了通过各种并行方式将模型、优化器状态等数据切分到不同设备外，还可以通过调整流水并行的负载偏置与重计算来精细调整内存的使用。
在实际大集群训练时，往往会限制global batch size，导致micro_batch_num开不大，产生较大的bubble。pp-interleave会将每个stage进一步划分为pp_interleave_num份mini stage，通过合理排布流水线降低bubble，从而提升端到端性能。

> 参考上文，pp-interleave 使用场景限制较大，只在pipeline stage >1 且micro batch num 无法开大(micro batch num==pipeline stage)的场景下（通常该场景下bubble较大）有效果，开启pp-interleave 时内存占用变大且增加额外通信，因此当bubble不大时会产生负收益，性能反而降低。

在mindformers/models/utils.py中提供了LayerSetting类，用于灵活配置每个stage包含的层数、重计算开启的层数、选择重计算的算子及开启的层数。

## 配置多流水交织

pp_interleave(virtual pipeline)官网配置介绍：[set_auto_parallel_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_auto_parallel_context.html?highlight=pipeline_interleave)
MindFormers中，开启多流水交织需要在parallel中配置，例如使用1f1b排布方式：

```yaml
parallel:
  pipeline_config:
    pipeline_interleave: True
    pipeline_scheduler: '1f1b'
```

之后在model_config中配置pp_interleave_num，例如开为2时：

```yaml
model:
  model_config:
    pp_interleave_num: 2
```

## 流水并行的负载均衡

流水并行默认网络层数num_layers可以被pp数pipeline_stage整除，每个stage中包含num_layers/pipeline_stage层。
如果网络层数num_layers不能被pp数pipeline_stage整除，或者调整每个stage中包含的层数，那么可以通过offset参数进行配置。

offset可以传入一个list或tuple，此时，list的元素个数需要等于pipeline_stage，list的各元素求和需要等于num_layers % pipeline_stage。
每个stage的层数则为n_layer_i = (num_layers // pipline_stage) + offset[i]

例如，一个网络有48层，pipeline_stage为5，offset设为[0,1,1,1,0]，那么这5个stage的层数为9,10,10,10,9。

当开启pp-interleave时，offset还可以传入一个二维list或tuple，内层的list表示在某个interleave中的pipeline_stage的负载偏置。
这种输入情况下，外层list的元素个数应等于pp_interleave_num，内层list的元素个数应等于pipeline_stage。
每个stage的层数则为n_layer_i_j = (num_layers // (pipline_stage * pp_interleave_num)) + offset[i][j]

例如，一个网络有48层，pp_interleave_num为2，pipeline_stage为5，offset设为[[0,1,1,1,1],[1,1,1,1,0]]，那么第一个interleave的5个stage的层数为4,5,5,5,5；第二个interleave的5个stage的层数为5,5,5,5,4。
在日志中会打印输入规范化后的层数与pp_id信息：

```log
INFO - num_layers per stage: [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]]
INFO - Accumulated num_layers per stage: [[4, 9, 14, 19, 24], [29, 34, 39, 44, 48]]
INFO - Pipeline id list: [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
INFO - Interleave id list: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

## 配置重计算与选择重计算

重计算可以显著降低训练时的激活内存，但会额外增加一些计算。

- **recompute**: （按层）完全重计算，可配置为bool，整数型的list或tuple或二维list或tuple。
    配置为bool类型时，对所有层开启或关闭完全重计算；
    配置为整数型list或tuple时，代表每个pipline_stage中有多少层开启完全重计算，pp_interleave_num>1时开启的重计算层数会均匀分配到各interleave中；
    配置为整数型二维list或tuple时，代表每个mini stage中有多少层开启完全重计算。
- **select_recompute**: （按算子）选择重计算，可配置为bool，整数型的list或tuple或二维list或tuple，字符串的list或tuple，以及字典，默认选择重计算算子为['feed_forward\\.mul', 'feed_forward\\.w1\\.activation\\.silu']。
    配置为bool类型时，对所有层开启或关闭默认算子的选择重计算；
    配置为整数型list或tuple时，代表每个pipline_stage中有多少层开启默认算子的选择重计算，pp_interleave_num>1时开启的选择重计算层数会均匀分配到各interleave中；
    配置为整数型二维list或tuple时，代表每个mini stage中有多少层开启默认算子的选择重计算。
    配置为字符串list或tuple时，代表对哪些算子开启选择重计算，算子名通过正则表达式匹配，层级关系通过'\\.'分割；
    配置为字典时，key值对应算子名，value值对应选择重计算的配置方式，这种配法可以对每个算子精细配置重计算策略；
- **select_comm_recompute**:（按算子）选择通信重计算，配置方式与**select_recompute**相同，默认选择通信重计算算子为['.*\\.norm']。一般仅对layer norm或类似层进行配置。
- **parallel_optimizer_comm_recompute**:优化器并行通信重计算,开启后在自动并行或半自动并行模式下，指定Cell内部由优化器并行引入的AllGather通信是否重计算。
- **mp_comm_recompute**:模型并行通信重计算,开启后在自动并行或半自动并行模式下，指定Cell内部由模型并行引入的通信操作是否重计算。
- **recompute_slice_activation**:切片重计算,是否对将保留在内存中的Cell输出进行切片。

注：1. 如果某一层同时配置了完全重计算与选择重计算，则按完全重计算生效。2. 在一维整数型list或tuple中的整数可以替换为True或False，代表

例如：一个网络有48层，pp_interleave_num为2，pipeline_stage为5，offset设为[[0,1,1,1,1],[1,1,1,1,0]]，重计算配置为如下

```yaml
# recompute config
recompute_config:
  recompute: [[2,1,0,0,0],[1,0,0,0,0]]
  select_recompute:
    'feed_forward\.w1\.activation\.silu': True
    'feed_forward\.mul': True
    'feed_forward\.w1\.matmul': [[1,0,0,0,0],[2,1,0,0,0]]
    'feed_forward\.w3\.matmul': [2,1,0,0,0]
  select_comm_recompute: ['ffn_norm\.norm','attention_norm\.norm']
```

在日志中会打印将输入格式规范化后的重计算策略信息：

```log
INFO - Formative layer_recompute: [[2, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
INFO - Formative select_recompute: {'feed_forward\.w1\.activation\.silu': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.mul': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.w1\.matmul': [[1, 0, 0, 0, 0], [2, 1, 0, 0, 0]], 'feed_forward\.w3\.matmul': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]}
INFO - Formative select_comm_recompute: {'ffn_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'attention_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]]}
```

随后会打印每一层重计算的配置方式。