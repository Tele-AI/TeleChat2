# DeepSeek-V3

## 模型描述

DeepSeek-V3是由DeepSeek（深度求索）推出的一个强大的专家混合（MoE）语言模型，它拥有671B总参数，其中激活参数量为37B。为了实现高效推理和低成本训练，DeepSeek-V3采用了多头潜注意力（MLA）和DeepSeekMoE架构，这在DeepSeek-V2中得到了充分验证。此外，DeepSeek-V3 还率先采用了无辅助损失的负载均衡策略，并设定了多token预测训练目标，以提高性能。DeepSeek-V3在14.8万亿个多种类的高质量token上进行预训练，接着通过监督微调和强化学习充分优化其能力。综合评估显示，在发布时DeepSeek-V3的性能优于其他开源模型，并可与领先的闭源模型相媲美。尽管性能卓越，DeepSeek-V3 的全部训练成本非常低，且其训练过程也非常稳定。

```text
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Haowei Zhang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Li and Hui Qu and J. L. Cai and Jian Liang and Jianzhong Guo and Jiaqi Ni and Jiashi Li and Jiawei Wang and Jin Chen and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Xu and Leyi Xia and Liang Zhao and Litong Wang and Liyue Zhang and Meng Li and Miaojun Wang and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingming Li and Ning Tian and Panpan Huang and Peiyi Wang and Peng Zhang and Qiancheng Wang and Qihao Zhu and Qinyu Chen and Qiushi Du and R. J. Chen and R. L. Jin and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runxin Xu and Ruoyu Zhang and Ruyi Chen and S. S. Li and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaoqing Wu and Shengfeng Ye and Shengfeng Ye and Shirong Ma and Shiyu Wang and Shuang Zhou and Shuiping Yu and Shunfeng Zhou and Shuting Pan and T. Wang and Tao Yun and Tian Pei and Tianyu Sun and W. L. Xiao and Wangding Zeng and Wanjia Zhao and Wei An and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and X. Q. Li and Xiangyue Jin and Xianzu Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaojin Shen and Xiaokang Chen and Xiaokang Zhang and Xiaosha Chen and Xiaotao Nie and Xiaowen Sun and Xiaoxiang Wang and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xinnan Song and Xinxia Shan and Xinyi Zhou and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and Y. K. Li and Y. Q. Wang and Y. X. Wei and Y. X. Zhu and Yang Zhang and Yanhong Xu and Yanhong Xu and Yanping Huang and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Li and Yaohui Wang and Yi Yu and Yi Zheng and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Ying Tang and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yu Wu and Yuan Ou and Yuchen Zhu and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yukun Zha and Yunfan Xiong and Yunxian Ma and Yuting Yan and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhen Huang and Zhen Zhang and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhipeng Xu and Zhiyu Wu and Zhongyu Zhang and Zhuoshu Li and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Ziyi Gao and Zizheng Pan},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## 模型文件

MindSpore Transformers中已提供DeepSeek-V3基于MindSpore的实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
     deepseek3/
      ├── deepseek2_model.py                  # DeepSeek-V2模型代码
      ├── deepseek2_config.py                 # DeepSeek-V2配置代码
      ├── deepseek3_model.py                  # DeepSeek-V3模型代码
      └── deepseek3_config.py                 # DeepSeek-V3配置代码
    ```

2. 模型配置：

    ```text
     deepseek3/
      ├── parallel_speed_up.json              # 数据集并行通信配置
      └── deepseek3_671b/
           ├── pretrain_deepseek3_671b.yaml   # 预训练任务配置
           └── finetune_deepseek3_671b.yaml   # 微调任务配置
    ```

3. 数据集处理脚本：

    ```text
    deepseek3/
      ├── wikitext_to_bin.py                 # wikitext数据预处理
      ├── deepseek3_conversation.py          # 微调chat_template实现
      └── deepseek3_preprocess.py            # alpaca数据预处理
    ```

## 模型权重下载

### 微调权重准备

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，`tokenizer.json`文件也在链接中下载。

| 模型名称                         |                                     Base权重（建议微调使用）                                      |                   Instruct权重（建议推理使用）                   |
|:-----------------------------|:---------------------------------------------------------------------------------------:|:------------------------------------------------------:|
| deepseek-ai/DeepSeek-V3-Base |               [Link](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)               | [Link](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| DeepSeek-V3-Base_4layer      | [Link](https://modelers.cn/models/mindformers-club/weights/tree/main/deepseekv3_4layer) |                                                        |

### 推理权重准备

用户可以从[魔乐社区](https://modelers.cn/models/MindSpore-Lab/DeepSeek-V3)下载权重进行推理，无需自己转换。

执行以下命令为自定义下载路径`./model_path`添加白名单：

```shell
export HUB_WHITE_LIST_PATHS=./model_path
```

执行以下 Python 脚本从魔乐社区下载昇思 MindSpore 版本的 DeepSeek-V3 文件至指定路径`./model_path`。下载的文件包含模型代码、权重、分词模型和示例代码，占用约 1.4TB 的磁盘空间：

```python
from openmind_hub import snapshot_download

snapshot_download(
    repo_id="MindSpore-Lab/DeepSeek-V3",
    local_dir="./model_path",
    local_dir_use_symlink=False
)
```

> 注意事项：
> - `./model_path` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 1.4TB）。
> - 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。

## 预训练

MindSpore Transformers支持对DeepSeek-V3进行预训练。仓库中提供了一份[预训练配置文件](#模型文件)供参考，该配置基于128台Atlas 800T A2（64G），使用Wikitext-2数据集进行预训练。为了方便体验，本章节基于此配置进行修改，缩小了DeepSeek-V3模型参数量，使其能够在单台Atlas 800T A2（64G）上拉起预训练流程。

### 环境准备

准备一台Atlas 800T A2（64G）训练服务器。MindSpore Transformers的环境依赖如下：

| Python | MindSpore |     CANN      |  固件与驱动   |
|:------:|:---------:|:-------------:|:--------:|
|  3.10  |  2.4.10   | 8.0.RC3.beta1 | 24.1.RC3 |

#### 安装固件与驱动

点击[此处](https://www.hiascend.com/hardware/firmware-drivers/community)下载固件与驱动的安装包，参考[昇腾官方教程](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html)进行安装。

> 固件与驱动需要安装24.1.RC3及以上版本，版本过低请进行升级

#### 准备Docker容器：

提供了DeepSeek-V3预训练专用Docker镜像（镜像中已包含CANN、MindSpore，无需手动安装），通过如下步骤进行软件环境搭建。

1. 下载Docker镜像

   使用如下命令下载DeepSeek-V3预训练专用镜像：

   ```bash
   docker pull swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.4.10-train:20250209
   ```

2. 基于镜像创建容器

   使用如下命令新建容器：

   ```bash
   image_name=swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.4.10-train:20250209
   docker_name=deepseek_v3
   docker run -itd -u root \
   --ipc=host --net=host \
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
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/bin/hccn_tool \
   -v /etc/ascend_install.info:/etc/ascend_install.info \
   -v /var/log/npu:/usr/slog \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /etc/hccn.conf:/etc/hccn.conf \
   --name "$docker_name" \
   "$image_name" \
   /bin/bash
   ```

3. 进入容器

   使用如下命令进入容器，并进入代码目录：

   ```bash
   docker exec -ti deepseek_v3 bash
   export MINDFORMERS_HOME=/home/work/mindformers
   cd $MINDFORMERS_HOME
   ```

#### 安装MindSpore Transformers

> 镜像中已经安装好了DeepSeek-V3预训练所需的MindSpore Transformers版本，如使用镜像可跳过此步骤。

执行如下命令拉取MindSpore Transformers代码，并编译安装：

```shell
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
git checkout e45eb7c5
bash build.sh
```

### 数据集准备

以Wikitext-2数据集为例，参考如下步骤将数据集处理成Megatron BIN格式文件。

1. 下载数据集和分词模型文件

   - 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

   - 分词模型下载：分词模型[tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

2. 生成Megatron BIN格式文件

   将数据集文件`wiki.train.tokens`和分词模型文件`tokenizer.json`放置在`../dataset`下

   使用以下命令将数据集文件转换为BIN格式文件。

   ```shell
   cd $MINDFORMERS_HOME
   python research/deepseek3/wikitext_to_bin.py \
   --input ../dataset/wiki.train.tokens \
   --output-prefix ../dataset/wiki_4096 \
   --vocab-file ../dataset/tokenizer.json \
   --seq-length 4096 \
   --workers 1
   ```

3. 构建Megatron BIN数据集模块

   执行如下命令构建Megatron BIN数据集模块。如使用提供的镜像请跳过此操作。

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

### 修改配置

修改预训练配置文件[pretrain_deepseek3_671b.yaml](#模型文件)，使其能够在单台Atlas 800T A2（64G）上运行，保存为`pretrain_deepseek3_1b.yaml`。以下仅列出修改项，其余配置与原文件保持一致。

1. 修改模型配置

   按照如下方式修改以缩小模型规模：

   ```yaml
   # model config
   model:
     model_config:
       hidden_size: 2048                                 # 修改为2048
       num_layers: &num_layers 3                         # 修改为3
       num_heads: 8                                      # 修改为8
       intermediate_size: 6144                           # 修改为6144
       offset: 0                                         # 修改为0
   ```

2. 修改MoE配置

   按照如下方式修改以缩小专家混合结构的规模：

   ```yaml
   #moe
   moe_config:
     expert_num: &expert_num 16                          # 修改为16
     first_k_dense_replace: 1                            # 修改为1
   ```

3. 修改并行配置

   缩小每种并行方式的切分数目，以适合在单台Atlas 800T A2（64G）上并行训练：

   ```yaml
   # parallel config for devices num=8
   parallel_config:
     data_parallel: &dp 2                                # 修改为2
     model_parallel: 2                                   # 修改为2
     pipeline_stage: 2                                   # 修改为2
     expert_parallel: 2                                  # 修改为2
     micro_batch_num: &micro_batch_num 4                 # 修改为4
   # parallel context config
   parallel:
     parallel_optimizer_config:
       optimizer_weight_shard_size: 8                    # 修改为8
   recompute_config:
     recompute: False                                    # 修改为False
   ```

4. 修改数据集配置

   配置数据集BIN文件路径：

   ```yaml
   # dataset
   train_dataset: &train_dataset
     data_loader:
       config:
         data_path:
           - "1"
           - "../dataset/wiki_4096_text_document"              # 修改此项为数据集BIN文件路径
   ```

   配置数据集并行通信配置路径：

   ```yaml
   # mindspore context init config
   context:
     ascend_config:
       parallel_speed_up_json_path: "./research/deepseek3/parallel_speed_up.json"  # 修改此项为数据集并行通信配置路径，需要固件与驱动版本不低于24.1.RC3
   ```

### 拉起任务

进入DeepSeek-V3代码目录并执行以下命令拉起单台Atlas 800T A2（64G）预训练任务：

```shell
export MS_DEV_DYNAMIC_SINK1=False
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--config research/deepseek3/deepseek3_671b/pretrain_deepseek3_1b.yaml"
```

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可查看训练状态（由于开启了流水并行`pipeline_stage: 2`，真实loss只显示在最后一张卡的日志`worker_7.log`中，其余卡均显示`loss`为`0`）：

```shell
tail -f ./output/msrun_log/worker_7.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

### 扩展：多机训练

如果服务器资源充足，可以参考如下方式拉起多台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`1023`。

```shell
master_ip=192.168.1.1
node_rank=0

cd $MINDFORMERS_HOME
export MS_DEV_DYNAMIC_SINK1=False
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--config research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml" \
1024 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。

如有关于DeepSeek-V3预训练的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。

## 全参微调

MindSpore Transformers支持对DeepSeek-V3进行全参微调。仓库中提供了一份[微调配置文件](#模型文件)供参考，该配置基于128台Atlas 800T A2（64G），使用alpaca数据集进行全参微调。为了方便体验，本章节基于此配置进行修改，缩小了DeepSeek-V3模型参数量，使其能够在4台Atlas 800T A2（64G）上拉起微调流程。

### 环境准备

参考[预训练-环境准备章节](#环境准备)

### 数据集准备

以[alpaca数据集](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)为例，参考如下步骤将数据集处理成Mindrecord格式文件。

  执行`research/deepseek3/deepseek3_preprocess.py`文件，进行数据预处理和Mindrecord数据生成。

  ```shell
  python research/deepseek3/deepseek3_preprocess.py \
   --dataset_type 'qa' \
   --input_glob /path/alpaca_data.json \
   --tokenizer_file /path/tokenizer.json \
   --seq_length 4096 \
   --output_file /path/alpaca-messages.mindrecord

  # 参数说明
  dataset_type:     预处理数据类型
  input_glob:       alpaca数据集原始文件路径
  tokenizer_file:   tokenizer.json文件路径
  seq_length:       输出数据的序列长度
  output_file:      输出文件的保存路径
  ```

### 模型权重准备

权重下载参考[模型权重下载](#模型权重下载)，体验demo可以下载DeepSeek-V3-Base_4layer，可以跳过[模型权重转换](#模型权重转换)步骤。

#### 模型权重转换

下载完成后，运行`research/deepseek3/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python research/deepseek3/convert_weight.py --torch_ckpt_path TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:            模型名称
torch_ckpt_path:  下载HuggingFace权重的文件夹路径
output_path:      转换后的MindSpore权重文件保存路径
dtype:            转换权重的精度
```

#### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

  从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。可以按照下列参考教程进行离线切分保存，也可以在运行时启用自动切分策略，后续[拉起任务等章节](#拉起任务)示例命令中采用运行时自动切分策略。

  通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

  以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

### 修改配置

修改微调配置文件[finetune_deepseek3_671b.yaml](#模型文件)，使其能够在4台Atlas 800T A2（64G）上运行，保存为`finetune_deepseek3_4layer.yaml`。此修改保留了模型的三种transformer_block层，分别为dense层、Moe层、MTP层。以下仅列出修改项，其余配置与原文件保持一致。

1. 修改模型配置

   ```yaml
   # model config
   model:
     model_config:
       num_layers: &num_layers 3                         # 修改为3
       offset: 0                                         # 修改为0
   ```

2. 修改MoE配置

   ```yaml
   #moe
   moe_config:
     first_k_dense_replace: 1                            # 修改为1
     use_gating_sigmoid: True
   ```

3. 修改并行配置

   ```yaml
   # parallel config for devices num=32
   parallel_config:
     data_parallel: &dp 4                                # 修改为4
     model_parallel: 4                                   # 修改为4
     pipeline_stage: 2                                   # 修改为2
     micro_batch_num: &micro_batch_num 8                 # 修改为8
   # parallel context config
   parallel:
     parallel_optimizer_config:
       optimizer_weight_shard_size: 4                    # 修改为4
   recompute_config:
     recompute: False                                    # 修改为False
   ```

4. 修改数据集配置

   配置数据集文件路径：

   ```yaml
   # dataset
   train_dataset: &train_dataset
     data_loader:
       dataset_dir: "./dataset"                # 修改此项为数据集mindrecord文件路径
   ```

### 拉起任务

进入mindformers根目录并执行以下命令拉起4台Atlas 800T A2（64G）微调任务：

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`3`。

```shell
master_ip=192.168.1.1
node_rank=0

bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--load_checkpoint /path/checkpoint_path \
--load_ckpt_format safetensors \
--output_dir ./output \
--auto_trans_ckpt True \
--config research/deepseek3/deepseek3_671b/finetune_deepseek3_4layer.yaml \
--run_mode finetune" \
32 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。
> load_checkpoint修改为原始权重路径，output_dir修改为用户想要保存训练后权重的路径。
> 如开启自动权重切分auto_trans_ckpt，load_checkpoint路径与output_dir路径需要是多机共享路径。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，在node_rank最后的机器使用以下命令可查看训练状态（由于开启了流水并行`pipeline_stage: 2`，真实loss只显示在最后一个stage的日志（worker_16.log ~ worker_31.log，建议使用最后一张卡的日志）中，其余卡显示`loss`为`0`）：

```shell
tail -f ./output/msrun_log/worker_31.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

### 扩展：整网微调

整网需要128台机器，在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`127`。

```shell
master_ip=192.168.1.1
node_rank=0

bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--load_checkpoint /path/checkpoint_path \
--load_ckpt_format safetensors \
--output_dir ./output \
--auto_trans_ckpt True \
--config research/deepseek3/deepseek3_671b/finetune_deepseek3_671b.yaml \
--run_mode finetune" \
1024 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。
> load_checkpoint修改为原始权重路径，output_dir修改为用户想要保存训练后权重的路径。
> 如开启自动权重切分auto_trans_ckpt，load_checkpoint路径与output_dir路径需要是多机共享路径。

如有关于DeepSeek-V3微调的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。

## 推理

MindSpore Transformers支持对DeepSeek-V3的推理。为了方便体验，仓库中提供了推理脚本，配置文件以及镜像，目前推理需要4台Atlas 800T A2（64G）机器。

### 准备容器

提供了DeepSeek-V3推理专用Docker镜像（镜像中已包含CANN、MindSpore，无需手动安装），通过如下步骤进行软件环境搭建。

1. 下载Docker镜像

   使用如下命令下载DeepSeek-V3推理专用镜像：

   ```bash
   docker pull swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.5.0-infer:20250209
   ```

2. 基于镜像创建容器

   使用如下命令新建容器：

   ```bash
   docker run -itd --privileged  --name=deepseek-v3-infer --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.5.0-infer:20250209 \
   bash
   ```

> 注意事项：

- 起容器时，如果有部分宿主机的hostname是一致的，需要在起容器的时候修改容器的hostname，保证所有容器的hostname都不一致。

3. 进入容器

   使用如下命令进入容器，并进入代码目录：

   ```bash
   docker exec -ti deepseek-v3-infer bash
   export MINDFORMERS_HOME=/home/work/mindformers
   export HCCL_OP_EXPANSION_MODE=AIV
   export MS_ENABLE_LCCL=off
   export EXPERIMENTAL_KERNEL_LAUNCH_GROUP="thread_num:2,kernel_group_num:8"
   export PYTHONPATH=$MINDFORMERS_HOME:$PYTHONPATH
   cd $MINDFORMERS_HOME
   ```

### 下载权重

权重下载参考[推理权重准备](#推理权重准备)，推理权重无需自己转换，可直接用于推理。

### 修改配置

仓库上提供的`research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml`中有部分配置需要根据实际进行修改，需要修改的地方如下：

`load_checkpoint`需要修改成权重存放的文件夹的绝对路径。

```yaml
load_checkpoint: "/path/to/deepseekv3/model.safetensors"
```

`vocab_file`和`tokenizer_file`需要修改成`tokenizer.json`的绝对路径。

```yaml
processor:
  tokenizer:
    vocab_file: '/path/to/deepseekv3/tokenizer.json'
    tokenizer_file: '/path/to/deepseekv3/tokenizer.json'
```

### 拉起推理任务

分别在4台机器上执行如下命令进行分布式推理，设置master_ip为IP地址，即Node 0服务器的IP。

在Node 0服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode predict \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--predict_data '请介绍一下北京的景点'" \
32 8 $master_ip 8888 0 output/msrun_log False 300
```

在Node 1服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode predict \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--predict_data '请介绍一下北京的景点'" \
32 8 $master_ip 8888 1 output/msrun_log False 300
```

在Node 2服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode predict \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--predict_data '请介绍一下北京的景点'" \
32 8 $master_ip 8888 2 output/msrun_log False 300
```

在Node 3服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode predict \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--predict_data '请介绍一下北京的景点'" \
32 8 $master_ip 8888 3 output/msrun_log False 300
```

预期的推理结果如下：

```txt
<｜begin▁of▁sentence｜>请介绍一下北京的景点\n\n北京，作为中国的首都，拥有丰富的历史文化遗产和众多的旅游景点。以下是一些著名的北京景点介绍：\n\n1. 故宫博物院：位于北京市中心，是中国明清两代的皇家宫殿，也是世界上现存规模最大、保存最为完整的木质结构古建筑群。故宫内收藏有大量珍贵的文物和艺术品，是了解中国古代皇家文化的重要窗口。\n\n2. 天安门广场：位于北京市中心，是世界上最大的城市广场之一。广场北侧是天安门城楼，南侧是人民英雄纪念碑，东侧是国家博物馆，西侧是人民大会堂。天安门广场是中国的象征，也是许多重要历史事件的发生地。\n\n3. 颐和园：位于北京市西北郊，是中国清朝时期的皇家园林，也是中国四大名园之一。颐和园以昆明湖和万寿山为基础，融合了江南园林的精致与北方园林的宏伟，是中国古典园林艺术的杰作。\n\n4. 长城：长城是中国古代的军事防御工程，横跨中国北部和中部地区。北京段的长城包括八达岭、慕田峪、金山岭等，是游客体验长城壮丽风光和了解中国古代军事文化的好去处。\n\n5. 天坛：位于北京市南部，是中国明清两代皇帝祭天祈谷的地方。天坛是中国古代祭天文化的代表，其建筑布局严谨，体现了中国古代的宇宙观和哲学思想。\n\n6. 圆明园：位于北京市西北郊，是清朝时期的皇家园林，曾被誉为“万园之园”。圆明园在1860年的第二次鸦片战争中被英法联军焚毁，现为遗址公园，是了解中国近代历史的重要场所。\n\n7. 北海公园：位于北京市中心，是中国现存最古老、保存最完整的皇家园林之一。北海公园以琼华岛为中心，湖光山色与古建筑相映成趣，是市民休闲娱乐的好去处。\n\n8. 北京奥林匹克公园：位于北京市朝阳区，是2008年北京奥运会的主要场馆区。公园内有鸟巢（国家体育场）、水立方（国家游泳中心）等标志性建筑，是体验现代体育文化和建筑艺术的好地方。\n\n9. 南锣鼓巷：位于北京市东城区，是北京最古老的街区之一，也是北京胡同文化的代表。南锣鼓巷以其独特的胡同风貌和丰富的文化底蕴，吸引了众多游客前来探访。\n\n10. 北京动物园：位于北京市西城区，是中国最早的动物园之一。北京动物园内饲养有众多珍稀动物，是了解中国动物多样性和进行科普教育的好地方。\n\n以上只是北京众多景点中的一部分，北京还有许多其他值得一游的地方，如798艺术区、雍和宫、景山公园等。每个景点都有其独特的历史背景和文化价值，值得游客深入探索。<｜end▁of▁sentence｜>
```

如有关于DeepSeek-V3推理的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。