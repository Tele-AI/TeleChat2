# 自动并行

自动并行模式让用户可以无需为网络中的每一个算子配置并行策略，即可达到高效并行训练的效果。当前MindSpore支持如下两种不同的自动并行方案：

- [切分策略传播算法(sharding_propagation)](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/sharding_propagation.html)：由少量使用shard接口配置并行策略的算子，向未配置的算子传播并行策略。在传播时，算法会尽量选取最少引发张量重排布通信的策略。
- [双递归策略搜索算法(recursive_programming)](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/sapp.html)：用户无需进行任何算子并行策略配置。其基于符号运算的代价模型可以自由适配不同的加速器集群，对于巨大网络以及大规模多卡切分能够保证快速生成最优策略。

详情参考官网关于[自动并行](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/auto_parallel.html)的说明。

## 使用说明

**当前本特性为实验性特性。**
当前MindFormers仓支持使用双递归策略搜索算法(recursive_programming)，进行自动化的并行策略生成。后续会新增支持切分策略传播算法(sharding_propagation)。目前验证过的模型如下所示：

|    模型    |    自动并行算法        |          参数量           |
| :--------: | :-------------------: | :-----------------------: |
|   LLaMA2   | recursive_programming | 7B<br>13B<br>70B<br>lora |
|  Baichuan2 | recursive_programming | 7B<br>13B<br>lora |
|    Qwen    | recursive_programming | 7B<br>14B<br>lora |
| PanguAlpha | recursive_programming | 2.6B<br>13B       |

在模型对应的configs路径下，提供了auto_parallel的yaml配置文件，用户可以快速启动单机多卡或多机多卡的自动并行训练。

自动并行模式在**mindspore r2.2版本及之后版本**生效，在**mindspore 2.1及之前版本**采用半自动并行模式。用户根据使用mindspore版本选择对应的yaml配置文件。自动并行模式配置文件带有"auto_parallel"后缀。

使用自动并行模式需要配置以下参数：

```yaml
use_parallel: True
parallel:
  parallel_mode: 2
  search_mode: "recursive_programming"
  auto_pipeline: True
parallel_config:
  mem_coeff: 0.1
```

- `use_parallel`: 启用并行模式的布尔值
- `parallel_mode`: 设置并行模式的整数值，其中0代表数据并行，1代表半自动并行，2代表自动并行
- `search_mode`: 设置搜索模式的字符串，可设置recursive_programming或者sharding_propagation

用户可以通过以下参数自定义自动并行的行为:

- `mem_coeff`: 正浮点值，表示高性能的数据并行策略（设置成较小值，例如<1）和低内存占用的模型并行策略（设置成较大值，例如>1）之间的权衡
- `auto_pipeline`: 启用自动流水线阶段数选择的布尔值

## batch_size

自动并行下的*batch_size*配置项和半自动并行下的*batch_size*略有不同。这是因为自动并行下不存在半自动并行的数据并行数（*data_parallel_num*）的概念，每个算子的数据并行和模型切分都不尽相同，没有统一的数据并行数和模型并行数。

| Pipeline | Full Batch | Global batch size formula |
| :------: | :--------: | :------------------------ |
| False    | False      | `batch_size * device_num * micro_batch_interleave_num * gradient_accumulation_steps` |
| False    | True       | `batch_size * data_parallel * micro_batch_interleave_num * gradient_accumulation_steps` |
| True     | False      | `batch_size * device_num * micro_batch_interleave_num * micro_batch_num` |
| True     | True       | `batch_size * data_parallel * micro_batch_interleave_num * micro_batch_num` |

可以看到自动并行下的*batch_size*实际上相当于半自动并行概念下的 `batch_size * data_parallel`

## mem_coeff

在 yaml 配置文件下新增了 mem_coeff 配置项，用来控制自动并行策略生成时，更倾向于数据并行或者模型并行。此配置项的默认值为0.1，更倾向于进行数据并行，但当模型参数量较大时，采用更多的数据并行会更可能出现内存不足的报错。此时，用户可以通过增大 mem_coeff 的值来控制自动并行策略生成更倾向于模型并行，mem_coeff 值越大，模型并行数越大（**建议用户以4为因数倍增**）。

**当前yaml配置文件下的mem_coeff配置值已经是最优，通常不需要用户进行调整。**

### 自定义mem_coeff

要分配一个合适的mem_coeff，需要对双递归策略搜索算法有一个粗略的了解。

1. 首先构建包含主要算子（例如矩阵乘法）的计算图。
2. 递归地为所有主要算子选择切分策略。
3. 最后将主要算子的切分策略传播到其他算子。

在步骤2中，会通过代价模型分析每一个切分策略的cost，最终选择cost最小的切分策略。以矩阵乘法算子为例，在矩阵乘法的切分策略上有四个选择，分别是切i轴、切j轴、切k轴和不切分：

$$
C_{ij} = \sum_k A_{ik} \cdot B_{kj}
$$

`mem_coeff`是代价模型的一个系数。
最高的值会引导代价模型偏向模型并行的切分策略（比如切j轴或切k轴），通常是最节省内存的。
较低的值会引导代价模型偏向数据并行的切分策略（比如切i轴），通常是最节省时间、性能最好的。
最低的值会引导代价模型偏向不切分的策略，这是最不节省内存但对于小算子可能是性能最好的。

`mem_coeff`只是代价模型中的一个系数，相同的值不能确保代价模型在所有场景下做出同样的选择。
默认值 0.1 通常有利于数据并行，但并不总是如此。
如果用户想要进行纯数据并行，那么可以通过调整`mem_coeff`达到效果。

### 指导教程

假设当前配置了 `mem_coeff = 0.01` ，自动并行根据代价模型为一部分算子生成了不切分的策略，用户想要所有算子都是纯数据并行策略。。

#### 1. 查看日志文件

在日志中搜索字符串 “Choose NOT to cut”，并查看代价模型中各个选择的cost。
如果没有出现这个字符串，则表明`mem_coeff`不会导致代价模型做出不切分策略的决策。

```text
If the I-axis is cut, the op-cost is 20.48, the rest-cost is 0, and the total cost is 20.48
If the K-axis is cut, the op-cost is 4096, the rest-cost is 0, and the total cost is 4096
If do NOT cut the axis, the op-cost is 16.7772, the rest-cost is 0, and the total cost is 16.7772
The costs of cutting the I-axis/J-axis/K-axis/no_cut are : [const vector]{20.48, 1.79749e+308, 4096, 16.772}
Choose NOT to cut
```

#### 2. 查看数据并行策略（切i轴）的总cost

在上面的例子中，代价模型切i轴的总$\newcommand{\costi}{\mathit{cost_i}}\costi$是20.48，其中算子cost是20.48，重排布cost是0.

#### 3. 查看不切分策略的总cost

在上面的例子中，代价模型不切分的总$\newcommand{\costr}{\mathit{cost_{rep}}}\costr$是16.7772，其中算子cost是16.7772，重排布cost是0.

#### 4. 根据公式计算出一个新mem_coef($\newcommand{\coeffnew}{\mathit{coeff_{new}}}\coeffnew$)，让代价模型切i轴的总cost更小，达到选择切i轴的目的

$$
\newcommand{\coeffold}{\mathit{coeff_{old}}}
\coeffnew = \frac{\costi \cdot \coeffold}{\costr} = \frac{20.48 \cdot 0.01}{16.7772} = 0.012207
$$

#### 5. 选择一个略高于新mem_coeff的数值，即`mem_coeff = 0.013`

## auto_pipeline

在 yaml 配置文件下新增了 `auto_pipeline` 配置项，用来决定是否由自动并行模式为流水线并行（pipeline stage）生成策略（pipeline stage number）。如果设置成True，那么用户无需配置流水线并行，自动并行模式会自动生成合适的流水线并行策略。如果设置为False，那么模型会执行用户配置的流水线并行，同时自动并行模式会在LOG中建议一个合适的流水线并行策略。请注意，自动并行功能生成的流水线并行策略（pipeline stage number）不会超过用户定义的`micro_batch_num`。

## Performance benchmarks

Comparison between performances of semi parallel versus SAPP.
The SpeedUp is given by the ratio of average "tokens per second" of auto parallel and average "tokens per second" of semi parallel. Hence a SpeedUp above 100% means that auto parallel runs faster than the semi parallel version.

| model | parallel_config in semi | mem_coeff in auto | SpeedUp |
|---|---|---|---|
| Baichuan2-7B | dp=8, mp=1, pp=1 | 0.01 | 97.4% |
| Baichuan2-13B | dp=8, mp=1, pp=1 | 0.01 | 98.0% |
| Qwen-7B | dp=2, mp=4, pp=1 | 4.5 | 96.2% |
| Qwen-14B | dp=1, mp=8, pp=1 | 4.5 |  111.3% |
| LLaMA2-7B | dp=8, mp=1, pp=1 | 0.1 | 101.2% |
| LLaMA2-13B | dp=8, mp=1, pp=1 | 0.1 | 99.8% |
| LLaMA2-70B | dp=2, mp=4, pp=8 | 0.1 | 93.5% |

These benchmarks have been carried out on a machine with 8 devices Ascend (except LLaMA2-70B which ran on 8 nodes, 64 devices) using the default .yaml files in the mindformers repository.
The software configuration is:

- MindSpore v2.3.0
- Mindformers 1.2.0

