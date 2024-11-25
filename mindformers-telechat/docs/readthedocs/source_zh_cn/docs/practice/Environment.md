# 环境变量使用说明

Mindformers提供了以下环境变量的配置说明，请根据使用场景自行配置使用。常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

## Mindformers

以下配置适用于mindformers框架相关的环境变量

| 环境变量                 | 功能       | 类型       | 取值                                                            | 说明                          |
|:---------------------|:---------|:---------|:--------------------------------------------------------------|:----------------------------|
| SHARED_PATHS         | 指定共享盘路径  | String   | 路径，支持相对路径与绝对路径，支持同时设置多个路径，<br/>如："/data/mount0,/data/mount1"。 | 设置后，会将指定的路径及其子路径视为共享路径      |
| DEVICE_NUM_PER_NODE  | 单机NPU数量  | Integer  | 单机实际NPU数量，不设置默认为8卡服务器。                                        |                             |
| CPU_AFFINITY         | CPU绑核    | String   | 1/0, 不设置默认为0                                                  | 设置后，将开启CPU绑核操作，可提升编译时间的稳定性  |

## 调试调优

以下配置适用于网络模型调试调优过程中的内存分析、DUMP功能、日志打印、通信等待等方面。

| 环境变量                        | 功能                  | 类型      | 取值                                                                                                                   | 说明                                                      |
|:----------------------------|:--------------------|:--------|:---------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------|
| LOG_MF_PATH                 | Mindformers日志保存位置   | String  | 路径，支持相对路径与绝对路径                                                                                                       | 设置后，会将Mindformers的日志文件保存到该路径，建议使用绝对路径。                  |
| MS_MEMORY_STATISTIC         | 内存析构                | Integer | 1：开启<br>   0：关闭<br> 默认值：0                                                                                            | 若开启内存析构，会在OOM时打印内存池占用情况。                                |
| MINDSPORE_DUMP_CONFIG       | 指定Dump功能所依赖的配置文件的路径 | String  | 文件路径，支持相对路径与绝对路径                                                                                                     |                                                         |
| GLOG_v                      | 控制Mindspore日志的级别    | Integer | 0-DEBUG<br> 1-INFO<br> 2-WARNING<br> 3-ERROR，表示程序执行出现报错，输出错误日志，程序可能不会终止<br> 4-CRITICAL，表示程序执行出现异常，将会终止执行程序<br> 默认值：2 | 指定日志级别后，将会输出大于或等于该级别的日志信息。                              |
| ASCEND_GLOBAL_LOG_LEVEL     | 控制CANN的日志级别         | Integer | 0-DEBUG<br> 1-INFO<br> 2-WARNING<br> 3-ERROR  <br> 默认值：3                                                             |                                                         |
| ASCEND_SLOG_PRINT_TO_STDOUT | 设置plog日志是否打屏        | Integer | 1：开启<br>   0：关闭<br> 默认值：0                                                                                            |                                                         |
| ASCEND_GLOBAL_EVENT_ENABLE  | 设置事件级别              | Integer | 1：开启Event日志<br>   0：关闭Event日志<br> 默认值：0                                                                              |                                                         |
| HCCL_EXEC_TIMEOUT           | HCCL进程执行同步等待时间      | Integer | 执行同步等待时间（s）<br> 默认值：1800s                                                                                            | 不同设备进程在分布式训练过程中存在卡间执行任务不一致的场景，通过该环境变量可控制设备间执行时的同步等待的时间。 |
| HCCL_CONNECT_TIMEOUT        | HCCL建链超时等待时间        | Integer | 建链等待时间（s）<br> 默认值：120s                                                                                               | 用于限制不同设备之间socket建链过程的超时等待时间。                            |
|PLOG_REDIRECT_TO_OUTPUT             |控制plog日志是否改变存储路径|bool|True:存储到./ouput目录下 <br> False: 默认存储位置 <br> 不添加该环境变量时，默认存储位置|设置之后方便用户查询plog日志|
|MF_LOG_SUFFIX                |设置所有log日志文件夹的自定义后缀|String|log文件夹的后缀 <br> 默认值：无后缀|添加一致的后缀，可以隔离各个任务的日志，不会被覆写|

## Ascend服务器相关配置

以下配置仅在Ascend服务器上适用。

| 环境变量                           | 功能                         | 类型      | 取值                                                                                | 说明                                                                    |
|:-------------------------------|:---------------------------|:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------|
| MS_GE_ATOMIC_CLEAN_POLICY      | 清理网络中atomic算子占用的内存的策略      | Integer | 0：集中清理网络中所有atomic算子占用的内存。<br>      1：不集中清理内存，对网络中每一个atomic算子进行单独清零。 <br> 默认值：1    |                                                                       |
| ENABLE_LAZY_INLINE             | 开启lazy inline              | Integer | 1：开启<br>   0：关闭<br> 默认值：1                                                         | 此特性在mindspore≥2.2.0下适用。通常在pipeline并行时使用以提高编译性能。默认开启，可配置关闭。            |
| ENABLE_LAZY_INLINE_NO_PIPELINE | 在非pipeline并行下开启lazy inline | Integer | 1：开启<br>   0：关闭<br> 默认值：0                                                         | lazy inline特性默认仅在pipeline并行模式下开启。如需在其他并行模式下使能lazy inline，可将该环境变量设置为1。 |
| MS_ASCEND_CHECK_OVERFLOW_MODE  | 溢出检测模式                     | String  | 默认：饱和模式，不设置此参数，当中间过程溢出时会上报，停止loss更新<br>    INFNAN_MODE：NAN模式，忽略过程中的溢出，结果非溢出就会继续训练 | 遇到持续溢出问题时可尝试设置此变量为INFNAN_MODE。                                        |

## Mindspore

mindspore相关环境变量请参考以下链接：

[MindSpore环境变量](https://www.mindspore.cn/docs/zh-CN/r2.2/note/env_var_list.html)
