# 前言

本指引旨在华为昇腾800TA2上运行TeleChat2，包含了相关素材的获取、环境的准备、模型的简单推理和微调。

# 环境准备

## 驱动固件环境准备

登陆服务器并查看固件驱动版本

```shell
npu-smi info -t board -i 0
npu-smi info
```

![产看驱动固件](../images/查看驱动固件.png)

若未安装过相关驱动或固件则提示无此命令

### 安装驱动固件

**安装包地址：**[Ascend-hdk-910b-npu_23.0.3_linux-aarch社区版](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.3/Ascend-hdk-910b-npu_23.0.3_linux-aarch64.zip?response-content-type=application/octet-stream)

> 下载上方的压缩包后，使用unzip直接解压
>
> 首次安装场景包括安装过驱动固件但是当前已卸载，按照 驱动 > 固件 的顺序安装
>
> 覆盖安装场景 按照 固件 > 驱动 的顺序安装

#### 前期准备

```sh
#执行如下命令增加执行权限和校验软件包的一致性和完整性。
chmod +x Ascend-hdk-910b-npu-driver_23.0.3_linux-aarch64.run
chmod +x Ascend-hdk-910b-npu-firmware_7.1.0.5.220.run
./Ascend-hdk-910b-npu-driver_23.0.3_linux-aarch64.run --check
./Ascend-hdk-910b-npu-firmware_7.1.0.5.220.run --check
#Debian系列（包含Ubuntu、Debian、UOS20、UOS20 SP1操作系统）推荐安装依赖
apt-get install -y dkms gcc 
```

![下载驱动和固件](../images/下载驱动和固件.png)

#### 从头安装驱动固件

```sh
#安装驱动
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all
#输出如下信息驱动安装成功：Driver package installed successfully!
#出现[ERROR]The list of missing tools: lspci,ifconfig，请安装缺少的依赖
apt-get install -y net-tools pciutils

#重启系统
reboot
#安装固件
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
#输出如下信息固件安装成功：Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect
#重启系统
reboot
#查看驱动加载是否成功
npu-smi info
```

### 卸载驱动固件

```sh
#卸载驱动
/usr/local/Ascend/driver/script/uninstall.sh

#卸载固件
/usr/local/Ascend/firmware/script/uninstall.sh
```

## 容器环境准备

### docker安装

```sh
dnf install  -y docker runc
sudo systemctl start docker
sudo docker images
```

![docker安装](../images/docker安装.png)

### 构建镜像

- 访问[ascend-mindspore](https://www.hiascend.com/developer/ascendhub/detail/9de02a1a179b4018a4bf8e50c6c2339)
  选择24.0.RC1-A2-openeuler20.03版本，获取登录访问权限

![基础镜像下载](../images/基础镜像下载.png)

- 使用[Dockerfile_TeleChat_ms](https://github.com/Tele-AI/TeleChat2/blob/main/Dockerfile_TeleChat_ms) 搭建镜像

```sh
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-mindspore:24.0.RC1-A2-openeuler20.03
docker build -f Dockerfile_TeleChat_ms -t telechat-ms:1.0 .
```

### 权重下载

- [telechat-7B-ms](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_7B/TeleChat2-7B_ms.ckpt)
- [telechat-35B-ms](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_35B/TeleChat2-35B_ms.tar)
- [telechat-115B-ms](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_115B/Telechat_115B.zip)

### 运行容器

```shell
docker run -itd -u 0 --ipc=host  --network host \
--name  telechat \
--privileged \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /data03:/mnt/model \
telechat-ms:1.0 \
/bin/bash
```

最后的模型挂载路径可以更换为本地保存模型的路径。

## 容器内环境配置

```shell
docker exec -it your_image_name bash
export PYTHONPATH=/workspace/TeleChat2/mindformers
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

验证mindspore

```sh
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```
![验证环境](../images/验证环境.png)

验证mindformers

```python
python -c "import mindformers;mindformers.run_check()"
```

# 模型推理

### 单卡推理

```sh
#进入工程目录
cd /workspace/TeleChat2/mindformers/research/telechat2
#运行推理
python3 run_telechat_predict.py  --vocab_file_path tokenizer.model  --checkpoint_path  /mnt/model/workspace/TeleChat2-7B_ms.ckpt --use_parallel False --yaml_file predict_telechat_7b.yaml

# 参数说明
vocab_file_path:tokenizer文件路径
checkpoint_path:模型权重文件路径
use_parallel:是否使用多卡推理
yaml_file:推理配置文件路径
```
![推理](../images/推理.png)

也可以自己传入输入文件推理，传入文件的参数为`--input_file`， 可以处理的格式示例：

```json
{"input": "生抽和老抽的区别？"}
{"input": "9.11和9.8哪个大"}
```

如果需要控制生成长度，可以：

1. 在配置文件中修改max_decode_length，该参数的含义是input_prompt + max_new_tokens的总tokens数量。

2. 在`run_telechat_predict.py`代码中调用generate前修改传入的max_new_tokens。

   注意：若添加max_new_tokens参数，该参数会覆盖上面的max_decode_length参数。


如果需要获取输出的logits 或者 scores，可以：

1. 在配置文件中设置return_dict_in_generate为True。

2. 设置output_scores、output_logits等参数为True。

   注意：修改后返回的数据类型是词典，需要对应改一下output的解码过程。

### 多卡推理

```shell
bash msrun_launcher.sh "python3 run_telechat_predict.py  --vocab_file_path tokenizer.model  --checkpoint_path  /mnt/model/workspace/TeleChat2-7B_ms.ckpt --use_parallel True --yaml_file predict_telechat_7b.yaml" 8
```

msrun_launcher.sh是并行需要的启动文件，所提供的参数如下表所示：

| **参数**         | **单机是否必选** | **多机是否必选** |    **默认值**    | **说明**                         |
| ---------------- | :--------------: | :--------------: | :--------------: | -------------------------------- |
| WORKER_NUM       |     &check;      |     &check;      |        8         | 所有节点中使用计算卡的总数       |
| LOCAL_WORKER     |        -         |     &check;      |        8         | 当前节点中使用计算卡的数量       |
| MASTER_ADDR      |        -         |     &check;      |    127.0.0.1     | 指定分布式启动主节点的ip         |
| MASTER_PORT      |        -         |     &check;      |       8118       | 指定分布式启动绑定的端口号       |
| NODE_RANK        |        -         |     &check;      |        0         | 指定当前节点的rank id            |
| LOG_DIR          |        -         |     &check;      | output/msrun_log | 日志输出路径，若不存在则递归创建 |
| JOIN             |        -         |     &check;      |      False       | 是否等待所有分布式进程退出       |
| CLUSTER_TIME_OUT |        -         |     &check;      |       7200       | 分布式启动的等待时间，单位为秒   |

抓取output/msrun_log/下任意worker的日志即可查看推理结果。

# 模型微调

处理示例数据

```sh
python telechat_preprocess.py --input_dataset_file /workspace/TeleChat2/datas/demo_tool.jsonl --vocab_file_path ./tokenizer.model --max_length 8192 --output_path mindrecords

# 参数说明:
input_dataset_file: 预训练的数据集
vocab_file_path: 词模型文件路径
max_length: 数据集长度
output_path: 生成数据集的路径
#微调数据集的格式为{"system":系统提示词,"dialog":{"role":(user or bot),"content":对话内容}}
#经过处理后的格式为<_system>system prompt<_user>content<_bot>content<_end>[126136, 29](代表'\n')
```

![数据处理](../images/数据处理.png)

### 单机多卡

开始微调

```shell
bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/workspace/TeleChat2-7B_ms.ckpt  --use_parallel True --auto_trans_ckpt True"  8 8 127.0.0.1 8118 0 output/msrun_log False 300
```

当控制台出现如下日志时：

```
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:0, log file:output/msrun_log/worker_0.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:1, log file:output/msrun_log/worker_1.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:2, log file:output/msrun_log/worker_2.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:3, log file:output/msrun_log/worker_3.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:4, log file:output/msrun_log/worker_4.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:5, log file:output/msrun_log/worker_5.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:6, log file:output/msrun_log/worker_6.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:7, log file:output/msrun_log/worker_7.log. Environment variable [RANK_ID] is exported.
```

说明启动微调成功。此时抓取每个worker的日志可以看到

![微调](../images/微调.png)

### 多机多卡

假设有两个服务器节点，节点0ip为192.168.1.1，作为主节点；节点1ip为192.168.1.2。每个节点8卡共16个卡。

首先在每张卡上设置环境变量

```sh
#For 192.168.1.1
export HCCL_IF_IP=192.168.1.1
unset RANK_TABLE_FILE

#For 192.168.1.2
export HCCL_IF_IP=192.168.1.2
unset RANK_TABLE_FILE
```

然后根据服务器节点数等信息，修改分布式策略，理论上dp\*mp\*sp\*pp == device_num。

```yaml
# 配置双机8卡分布式策略，以dp=8,mp=2,pp=1为例
parallel_config:
  data_parallel: 8
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
```

#### 自动权重转换

涉及到权重转换的详细教程请参考特性文档模型权重切分与合并。若模型权重在服务器共享盘上，可以尝试使用自动权重转换。

```sh
bash msrun_launcher.sh "python run_telechat.py \
 --config finetune_telechat_7b.yaml \
 --load_checkpoint /mnt/model/workspace/TeleChat2-7B_ms.ckpt \
 --train_dataset ./mindrecords \
 --use_parallel True \
 --auto_trans_ckpt True \ "
  16 8 192.168.1.1 8118 0 output/msrun_log False 300

#节点0与节点1启动命令仅参数NODE_RANK不同
bash msrun_launcher.sh "python run_telechat.py \
 --config finetune_telechat_7b.yaml \
 --load_checkpoint /mnt/model/workspace/TeleChat2-7B_ms.ckpt \
 --train_dataset ./mindrecords \
 --use_parallel True \
 --auto_trans_ckpt True \ "
  16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

#### 离线权重转换

- step 1. 打开策略文件保存开关

修改微调配置文件`finetune_telechat_7b.yaml`，将`only_save_strategy`设置为True。

- step 2 启动训练，注意关闭自动切分`auto_trans_ckpt`参数。

```sh
bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/workspace/TeleChat2-7B_ms.ckpt  --use_parallel True " 16 8 192.168.1.1 8118 0 output/msrun_log False 300

bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/workspace/TeleChat2-7B_ms.ckpt  --use_parallel True "  16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

![策略文件生成](../images/策略文件生成.png)

各节点的策略文件保存在各自的`output/strategy`目录下。

- step 3 离线权重转换 

在每台物理机上执行

```sh
python ../../mindformers/tools/ckpt_transorm/transform_checkpoint.py \
--src_checkpoint=/mnt/model/workspace/TeleChat2-7B_ms.ckpt \
--dst_checkpoint=/mnt/model/workspace/TeleChat2-7B-2mp8dp \
--dst_strategy=./output/strategy
```

转换完成后在`dst_checkpoint`目录下生成切片权重，注意此时会根据前缀生成新一级目录。

![权重切片](../images/权重切片.png)

- step 4 将配置文件修改回来

修改微调配置文件`finetune_telechat_7b.yaml`，将`only_save_strategy`设置为False。

- step 5 启动任务，启动命令同step2，此处传入的权重路径model_dir应该是按照model_dir/rank_x/xxx.ckpt格式存放

```bash
bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/workspace/TeleChat2-7B-2mp8dp/TeleChat2-7B_ms  --use_parallel True " 16 8 192.168.1.1 8118 0 output/msrun_log False 300

bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/workspace/TeleChat2-7B-2mp8dp/TeleChat2-7B_ms  --use_parallel True " 16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

### 微调权重合并

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python ../../mindformers/tools/ckpt_transorm/transform_checkpoint.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix telechat_7B
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

