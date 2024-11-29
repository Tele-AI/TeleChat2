# 前言

本指引旨在华为昇腾800TA2上运行TeleChat2，包含了相关素材的获取、环境的准备、模型的简单推理、模型的快速

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

**
安装包地址：**[Ascend-hdk-910b-npu_23.0.3_linux-aarch社区版](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.3/Ascend-hdk-910b-npu_23.0.3_linux-aarch64.zip?response-content-type=application/octet-stream)

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
```sh
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

# 模型推理

```sh
#进入工程目录
cd /workspace/TeleChat2/mindformers/research/telechat2
#运行推理
python3 run_telechat_predict.py  --vocab_file_path tokenizer.model  --checkpoint_path  /mnt/model/TeleChat2-7B_ms.ckpt --use_parallel False --yaml_file predict_telechat_7b.yaml

```
![推理](../images/推理.png)

# 模型微调

处理示例数据

```sh
python telechat_preprocess.py --input_dataset_file /workspace/TeleChat2/datas/demo_tool.jsonl --vocab_file_path ./tokenizer.model --max_length 8192 --output_path mindrecords

# 参数说明
# input_dataset_file: 预训练的数据集
# vocab_file_path: 词模型文件路径
# max_length: 数据集长度
# output_path: 生成数据集的路径
```

![数据处理](../images/数据处理.png)

开始微调

```shell
bash msrun_launcher.sh "python run_telechat.py  --config finetune_telechat_7b.yaml  --train_dataset ./mindrecords   --load_checkpoint /mnt/model/TeleChat2-7B_ms.ckpt  --use_parallel True --auto_trans_ckpt True"  8 8 127.0.0.1 8118 0 output/msrun_log False 300
```

