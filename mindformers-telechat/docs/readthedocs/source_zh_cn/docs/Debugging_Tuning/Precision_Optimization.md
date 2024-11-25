# 精度调优

## [Dump使用方法](https://mindspore.cn/tutorials/experts/zh-CN/r2.0/debug/dump.html)

异步/同步 dump的操作步骤基本一样，区别在于json文件不同。

**异步dump(训练结束后dump数据)**

1. 创建json配置文件

JSON文件的名称和位置可以自定义设置。

```json
{
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "/absolute_path",
        "net_name": "ResNet50",
        "iteration": "0|5-8|100-120",
        "saved_data": "tensor",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0,1,2,3,4,5,6,7],
        "op_debug_mode": 0,
        "file_format": "npy"
    }
}
```

- `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据或算子类型数据；设置成2，表示Dump脚本中通过`set_dump`指定的算子数据，`set_dump`的使用详见[mindspore.set_dump](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.set_dump.html) 。开启溢出检测时，此字段的设置失效，Dump只会保存溢出节点的数据。
- `path`：Dump保存数据的绝对路径。
- `net_name`：自定义的网络名称，例如：”ResNet50”。
- `iteration`：指定需要Dump的迭代。类型为str，用“|”分离要保存的不同区间的step的数据。如”0|5-8|100-120”表示Dump第1个，第6个到第9个， 第101个到第121个step的数据。指定“all”，表示Dump所有迭代的数据。PyNative模式开启溢出检测时，必须设置为”all”。
- `saved_data`: 指定Dump的数据。类型为str，取值成”tensor”，表示Dump出完整张量数据；取值成”statistic”，表示只Dump张量的统计信息；取值”full”代表两种都要。异步Dump统计信息只有在`file_format`设置为`npy`时可以成功，若在`file_format`设置为`bin`时选”statistic”或”full”便会错误退出。默认取值为”tensor”。
- `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。
- `kernels`：该项可以配置两种格式：
  1. 算子的名称列表。开启IR保存开关`set_context(save_graphs=2)`并执行用例，从生成的IR文件`trace_code_graph_{graph_id}`中获取算子名称。详细说明可以参照教程：[如何保存IR](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/debug/mindir.html#如何保存ir)。 需要注意的是，是否设置`set_context(save_graphs=2)`可能会导致同一个算子的id不同，所以在Dump指定算子时要在获取算子名称之后保持这一项设置不变。或者也可以在Dump保存的`ms_output_trace_code_graph_{graph_id}.ir`文件中获取算子名称，参考同步Dump数据对象目录。
  2. 还可以指定算子类型。当字符串中不带算子scope信息和算子id信息时，后台则认为其为算子类型，例如：”conv”。算子类型的匹配规则为：当发现算子名中包含算子类型字符串时，则认为匹配成功（不区分大小写），例如：”conv” 可以匹配算子 “Conv2D-op1234”、”Conv3D-op1221”。
- `support_device`：支持的设备，默认设置成0到7即可；在分布式训练场景下，需要dump个别设备上的数据，可以只在`support_device`中指定需要Dump的设备Id。
- `op_debug_mode`：该属性用于算子溢出调试，设置成0，表示不开启溢出；设置成1，表示开启AiCore溢出检测；设置成2，表示开启Atomic溢出检测；设置成3，表示开启全部溢出检测功能。在Dump数据的时候请设置成0，若设置成其他值，则只会Dump溢出算子的数据。
- `file_format`: dump数据的文件类型，只支持`npy`和`bin`两种取值。设置成`npy`，则dump出的算子张量数据将为host侧格式的npy文件；设置成`bin`，则dump出的数据将为device侧格式的protobuf文件，需要借助转换工具进行处理，详细步骤请参考异步Dump数据分析样例。默认取值为`bin`。

2. 设置Dump环境变量

```shell
export MINDSPORE_DUMP_CONFIG=${Absolute path of data_dump.json}
```

如果Dump配置文件没有设置`path`字段或者设置为空字符串，还需要配置环境变量`MS_DIAGNOSTIC_DATA_PATH`。

```shell
export MS_DIAGNOSTIC_DATA_PATH=${yyy}
```

则“$MS_DIAGNOSTIC_DATA_PATH/debug_dump”就会被当做`path`的值。若Dump配置文件中设置了`path`字段，则仍以该字段的实际取值为准。

- 在网络脚本执行前，设置好环境变量；网络脚本执行过程中设置将会不生效。
- 在分布式场景下，Dump环境变量需要在调用`mindspore.communication.init`之前配置。

3. [启动网络训练脚本](../start/Quick_Tour.md#方式一使用已有脚本启动)

可以在训练脚本中设置`set_context(reserve_class_name_in_scope=False)`，避免Dump文件名称过长导致Dump数据文件生成失败。

4. 解析Dump数据文件

可以使用MindSpore Insight的离线调试器来分析。离线调试器的使用方法详见[使用离线调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/debugger_offline.html) 。

**同步dump(训练的同时进行数据dump)**

参考[mindspore官网dump说明](https://mindspore.cn/tutorials/experts/zh-CN/r2.0/debug/dump.html)
