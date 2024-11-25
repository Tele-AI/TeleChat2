# 动态组网分布式启动

MindSpore在2.2.0版本之后提供了[动态组网方式启动分布式任务的教程](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/dynamic_cluster.html)

动态组网方式相比原分布式启动方式，最大优点在于无需用户提前准备rank_table，MindSpore框架将在内部完成多卡的通信调度，能够简化分布式任务的调度流程

MindFormers提供了动态组网启动的样例脚本 `scripts/run_distribute_ps_auto.sh`，基于run_mindformer.py脚本进行分布式任务的拉起，以下介绍使用动态组网脚本拉起分布式任务的流程

动态组网启动相关环境变量，run_distribute_ps_auto.sh 脚本中使用到：

- `SERVER_ID`: 当前服务器节点的ID，首节点为0，其余节点依次增加
- `SERVER_NUM`: 使用的服务器数量，单机为1，多机为具体服务器数
- `PER_DEVICE_NUMS`: 每台服务器使用的卡数，默认为8，可修改该环境变量以适配单机4卡等场景
- `MS_SCHED_HOST`: 调度进程的host ip，动态组网存在一个调度进程和数量等于卡数个的工作进程，调度进程与工作进程间的通信需指定ip与端口；指定首个服务器节点的ip地址即可
- `MS_SCHED_PORT`: 调度进程的端口号，不指定时使用sh脚本中的默认值，需注意不能存在端口冲突

shell脚本启动入参：

- `CONFIG_PATH`: 任务所需的模型配置文件路径
- `RUN_MODE`: 指定run_mindformer脚本的运行模式

以单机4卡运行gpt2模型训练为例：

step 1. 设置动态组网所需环境变量

```bash
# 单机，编号从0开始
export SERVER_ID=0
# 单机，数量为1
export SERVER_NUM=1
# 设置使用4卡
export PER_DEVICE_NUMS=4
# 设置为当前服务器ip
export MS_SCHED_HOST=xx.xx.xx.xx
```

step 2. 运行动态组网脚本，启动分布式训练

参考gpt2模型中的[预训练部分](../model_cards/gpt2.md#预训练)，进行数据集的准备与配置修改

使用以下命令以启动多卡训练：

```bash
cd scripts
bash run_distribute_ps_auto.sh /path/to/pretrain_gpt2_small_fp16.yaml train
```
