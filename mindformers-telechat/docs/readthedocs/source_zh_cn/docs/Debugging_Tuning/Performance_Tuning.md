# 性能调优

## [Profiler](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.Profiler.html?&highlight=profiler)数据采集

MindSpore中提供了profiler接口，可以对神经网络的性能进行采集。目前支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

mindformers将此功能集成到了套件内部，用户仅需修改配置文件中相应部分即可开启采集。

在配置文件中添加Profile相关参数：

```yaml
# profile
profile: False
profile_start_step: 1
profile_stop_step: 10
init_start_profile: True
profile_communication: True
profile_memory: True
```

- **profile_start_step** - 表示开始收集算子性能数据的step。默认值：1。
- **profile_stop_step** - 表示结束收集算子性能数据的step。默认值：10。
- **output_path** - 表示输出数据的路径。默认值："./output/profile/rank_{id}/profiler/"。
- **init_start_profile** - 该参数控制是否在Profiler初始化的时候开启数据采集。开启后profile_start_step将不生效。如果需要收集多设备通信数据则必须开启。默认值：True。
- **profile_communication** - (仅限Ascend) 表示是否在多设备训练中收集通信性能数据。当值为True时，收集这些数据。在单台设备训练中，该参数的设置无效。默认值：False。
- **profile_memory** - (仅限Ascend) 表示是否收集Tensor内存数据。当值为True时，收集这些数据。默认值：False。

修改配置文件后，[启动训练](../start/Quick_Tour.md#方式一使用已有脚本启动)，在输出目录下会生成相应的profile文件夹存放各rank的profiler文件。

```index
output
  ├─checkpoint
  └─profile
    ├─rank0
    │   └─profiler
    ├─...
    └─rank7
        └─profiler
......
```

## [Summary](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.SummaryCollector.html)数据采集

> ### 使用约束
>
> 使用SummaryMonitor时，如果使用Trainer高阶接口启动任务，则**必须**将代码放置到 `if __name__ == “__main__”` 中运行。

在配置文件的callbacks中添加SummaryMonitor和相关参数：

```yaml
callbacks:
  * type: SummaryMonitor
    summary_dir: None
    collect_freq: 10
    collect_specified_data: None
    keep_default_action: True
    custom_lineage_data: None
    collect_tensor_freq: None
    max_file_size: None
    export_options: None
```

参数

- **summary_dir** (str) - 收集的数据将存储到此目录。如果目录不存在，将自动创建。默认值："./output/summary/rank_{id}/"。
- **collect_freq** (int) - 设置数据收集的频率，频率应大于零，单位为 step 。如果设置了频率，将在(current steps % freq)=0时收集数据，并且将总是收集第一个step。需要注意的是，如果使用数据下沉模式，单位将变成 epoch 。不建议过于频繁地收集数据，因为这可能会影响性能。默认值：10。
- **collect_specified_data** (Union[None, dict]) - 对收集的数据进行自定义操作。您可以使用字典自定义需要收集的数据类型。例如，您可以设置{‘collect_metric’:False}不去收集metrics。支持控制的数据如下。默认值：None，收集所有数据。
    - **collect_metric** (bool) - 表示是否收集训练metrics，目前只收集loss。把第一个输出视为loss，并且算出其平均数。默认值：True。
    - **collect_graph** (bool) - 表示是否收集计算图。目前只收集训练计算图。默认值：True。
    - **collect_train_lineage** (bool) - 表示是否收集训练阶段的lineage数据，该字段-将显示在MindInsight的 [lineage页面](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/lineage_and_scalars_comparison.html) 上。默认值：True。
    - **collect_eval_lineage** (bool) - 表示是否收集评估阶段的lineage数据，该字段将显示在MindInsight的lineage页面上。默认值：True。
    - **collect_input_data** (bool) - 表示是否为每次训练收集数据集。目前仅支持图像数据。如果数据集中有多列数据，则第一列应为图像数据。默认值：True。
    - **collect_dataset_graph** (bool) - 表示是否收集训练阶段的数据集图。默认值：True。
    - **histogram_regular** (Union[str, None]) - 收集参数分布页面的权重和偏置，并在MindInsight中展示。此字段允许正则表达式控制要收集的参数。不建议一次收集太多参数，因为这会影响性能。注：如果收集的参数太多并且内存不足，训练将会失败。默认值：None，表示只收集网络的前五个超参。
    - **collect_landscape** (Union[dict, None]) - 表示是否收集创建loss地形图所需要的参数。如果设置None，则不收集任何参数。默认收集所有参数并且将会保存在 {summary_dir}/ckpt_dir/train_metadata.json 文件中。
        - **landscape_size** (int) - 指定生成loss地形图的图像分辨率。例如：如果设置为128，则loss地形图的分辨率是128*128。注意：计算loss地形图的时间随着分辨率的增大而增加。默认值：40。可选值：3-256。
        - **unit** (str) - 指定训练过程中保存checkpoint时，下方参数 intervals 以何种形式收集模型权重。例如：将 intervals 设置为[[1, 2, 3, 4]]，如果 unit 设置为step，则收集模型权重的频率单位为step，将保存1-4个step的模型权重，而 unit 设置为epoch，则将保存1-4个epoch的模型权重。默认值：step。可选值：epoch/step。
        - **create_landscape** (dict) - 选择创建哪种类型的loss地形图，分为训练过程loss地形图（train）和训练结果loss地形图（result）。默认值：{“train”: True, “result”: True}。可选值：True/False。
        - **num_samples** (int) - 创建loss地形图所使用的数据集的大小。例如：在图像数据集中，您可以设置 num_samples 是128，这意味着将有128张图片被用来创建loss地形图。注意：num_samples 越大，计算loss地形图时间越长。默认值：128。
        - **intervals** (List[List[int]]) - 指定loss地形图的区间。例如：如果用户想要创建两张训练过程的loss地形图，分别为1-5epoch和6-10epoch，则用户可以设置[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]。注意：每个区间至少包含3个epoch。
- **keep_default_action** (bool) - 此字段影响 collect_specified_data 字段的收集行为。True：表示设置指定数据后，其他数据按默认设置收集。False：表示设置指定数据后，只收集指定数据，不收集其他数据。默认值：True。
- **custom_lineage_data** (Union[dict, None]) - 允许您自定义数据并将数据显示在MindInsight的lineage页面上。在自定义数据中，key支持str类型，value支持str、int和float类型。默认值：None，表示不存在自定义数据。
- **collect_tensor_freq** (Optional[int]) - 语义与 collect_freq 的相同，但仅控制TensorSummary。由于TensorSummary数据太大，无法与其他summary数据进行比较，因此此参数用于降低收集量。默认情况下，收集TensorSummary数据的最大step数量为20，但不会超过收集其他summary数据的step数量。例如，给定 collect_freq=10 ，当总step数量为600时，TensorSummary将收集20个step，而收集其他summary数据时会收集61个step。但当总step数量为20时，TensorSummary和其他summary将收集3个step。另外请注意，在并行模式下，会平均分配总的step数量，这会影响TensorSummary收集的step的数量。默认值：None，表示要遵循上述规则。
- **max_file_size** (Optional[int]) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，如果不大于4GB，则设置 max_file_size=4*1024**3 。默认值：None，表示无限制。
- **export_options** (Union[None, dict]) - 表示对导出的数据执行自定义操作。注：导出的文件的大小不受max_file_size 的限制。您可以使用字典自定义导出的数据。例如，您可以设置{‘tensor_format’:’npy’}将tensor导出为 npy 文件。支持控制的数据如下所示。默认值：None，表示不导出数据。
    - **tensor_format** (Union[str, None]) - 自定义导出的tensor的格式。支持[“npy”, None]。默认值：None，表示不导出tensor。
        - **npy** - 将tensor导出为NPY文件。

训练完成后，summary日志文件将会输出至指定目录，目录结构如下：

```tree
└─summary
  └─rank_0
        events.out.events.summary.1596869898.hostname_MS
        events.out.events.summary.1596869898.hostname_lineage
```

## [MindInsight](https://mindspore.cn/mindinsight/docs/zh-CN/r2.0/index.html)可视化调优

通过MindInsight可以完成训练可视、性能调优、精度调优等任务。

训练可视功能主要包括训练看板、模型溯源、数据溯源等功能，训练看板中又包括标量、参数分布图、计算图、数据图、数据抽样、张量等子功能。

训练完成数据收集后，启动MindInsight，即可可视化收集到的数据。启动MindInsight时， 需要通过 `--summary-base-dir` 参数指定summary日志文件目录。MindInsight启动的默认端口为`8080`，当默认端口被占用时，可以通过`--port`参数指定端口。

通过命令行启动mindinsight
**可视化训练结果**

```shell
# 开启 summary-base-dir指定到mindformers生成的output/summary
mindinsight start --port 8000 --summary-base-dir xx/mindformers/output/summary
```

**可视化profiler性能数据**

```shell
# 开启 summary-base-dir指定到mindformers生成的output/profiler
mindinsight start --port 8000 --summary-base-dir xx/mindformers/output/profiler
```

启动成功后，通过浏览器访问 `http://127.0.0.1:8080` 地址，即可查看可视化页面。

如果要远程访问物理机上启动的可视化页面时，需要修改MindInsight的Host地址。首先通过`pip show mindinsight`获取MindInsight安装位置，打开安装目录下的`./conf/constants.py` ，修改其中的`Host`字段为`0.0.0.0`。

**关闭mindinsight**

```shell
#关闭
mindinsight stop --port 8000
```

## 保存IR图(中间编译图)

中间编译图又称为中间表示（IR），是程序编译过程中介于源语言和目标语言之间的程序表示，以方便编译器进行程序分析和优化，因此IR的设计需要考虑从源语言到目标语言的转换难度，同时考虑程序分析和优化的易用性和性能。

MindIR是一种基于图表示的函数式IR，其最核心的目的是服务于自动微分变换。自动微分采用的是基于函数式编程框架的变换方法，因此IR采用了接近于ANF函数式的语义。此外，借鉴Sea of Nodes和Thorin的优秀设计，采用了一种基于显性依赖图的表示方式。关于ANF-IR的具体介绍，可以参考[MindSpore IR文法定义](https://www.mindspore.cn/docs/zh-CN/r2.0/design/mindir.html#文法定义)。

MindSpore提供了保存中间编译图的功能，在使用MindFormers编写模型时，在图模式`set_context(mode=GRAPH_MODE)`下，可以通过设置MindSpore上下文来打开存图功能。

在MindSpore 1.10版本中，支持使用`set_context(save_graphs=True)`来打开存图功能。运行时会输出一些图编译过程中生成的一些中间文件。

在MindSpore 2.0版本中，若配置中设置了`set_context(save_graphs=1)`，运行时会输出一些图编译过程中生成的一些中间文件。当需要分析更多后端流程相关的ir文件时，可以设置`set_context(save_graphs=True)` 或`set_context(save_graphs=2)`。当需要更多进阶的信息比如可视化计算图，或者更多详细前端ir图时，可以设置`set_context(save_graphs=3)`。

可以通过`set_context(save_graphs_path='{path}')`来指定图保存路径。

当前主要有三种格式的IR文件：

- ir后缀结尾的IR文件：一种比较直观易懂的以文本格式描述模型结构的文件，可以直接用文本编辑软件查看。
- 通过设置环境变量`export MS_DEV_SAVE_GRAPTHS_SORT_MODE=1`可以生成的ir后缀结尾的IR文件：格式跟默认的ir文件基本相同，但是生成的异序ir文件的图的生成与打印顺序和默认ir文件不同。
- dot后缀结尾的IR文件：描述了不同节点间的拓扑关系，可以用[graphviz](http://graphviz.org/)将此文件作为输入生成图片，方便用户直观地查看模型结构。对于算子比较多的模型，推荐使用可视化组件[MindInsight](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/dashboard.html#计算图可视化)对计算图进行可视化。
