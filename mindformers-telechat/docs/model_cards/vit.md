# ViT

## 模型描述

ViT：全名Vision Transformer，不同于传统的基于CNN的网络结果，是基于Transformer结构的CV网络，2021年谷歌研究发表网络，在大数据集上表现了非常强的泛化能力。大数据任务（如：CLIP）基于该结构能有良好的效果。MindFormers提供的ViT权重及精度均是是基于MAE预训练ImageNet-1K数据集进行微调得到。

[论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F2010.11929): Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.

## 模型性能

- 基于Atlas 800

|                            config                            |         task         |  Datasets   |    metric     | score  | [train performance](#预训练) | [prediction performance](#推理) |
| :----------------------------------------------------------: | :------------------: | :---------: | :-----------: | :----: | :--------------------------: | :-----------------------------: |
| [vit_base_p16](../../configs/mae/run_vit_base_p16_224_100ep.yaml) | image_classification | ImageNet-1K | Top1-Accuracy | 0.8372 |      262.31 samples/s/p      |          363.50 (fps)           |

## 仓库介绍

`ViT` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/vit`

   ```bash
   model
       ├── __init__.py
       ├── convert_weight.py         # 权重转换脚本
       ├── vit.py                    # 模型实现
       ├── vit_config.py             # 模型配置项
       ├── vit_modules.py            # 模型所需模块
       └── vit_processor.py          # Model预处理
   ```

2. 模型配置：`configs/vit`

   ```bash
   model
       └── run_vit_base_p16_224_100ep.yaml         # vit_base模型启动配置
   ```

## 前期准备

### [mindformers安装](path/to/README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 多机RANK_TABLE_FILE合并(多机多卡必备环)

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

RANK_TABLE_FILE 双机16卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换

如果无需加载权重，或者使用from_pretrained功能自动下载，可以跳过此章节。

MindFormers提供高级接口from_pretrained功能直接下载MindFormerBook中的[vit_base_p16.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/vit/vit_base_p16.ckpt)，无需手动转换。

本仓库中的`vit_base_p16`来自于facebookresearch/mae的[ViT-Base](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth), 如需手动下载权重，可参考以下示例进行转换：

1. 从上述链接中下载`ViT-Base`的模型权重

2. 执行转换脚本，得到转换后的输出文件`vit_base_p16.ckpt`

```bash
python mindformers/models/vit/convert_weight.py --torch_path "PATH OF ViT-Base.pth" --mindspore_path "SAVE PATH OF vit_base_p16.ckpt"
```

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/vit`

```python
import mindspore
from mindformers import AutoModel, AutoConfig
from mindformers.tools.image_tools import load_image
from mindformers import ViTImageProcessor

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 模型标志加载模型
model = AutoModel.from_pretrained("vit_base_p16")

#模型配置加载模型
config = AutoConfig.from_pretrained("vit_base_p16")
# {'patch_size': 16, 'in_chans': 3, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4,
# ..., 'batch_size': 32, 'image_size': 224, 'num_classes': 1000}
model = AutoModel.from_config(config)

img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
image_processor = ViTImageProcessor(size=224)
processed_img = image_processor(img)

predict_result = model(processed_img)
print(predict_result)

# output
# (Tensor(shape=[1, 1000], dtype=Float32, value=
# [[-5.38996577e-01, -2.30418444e-02,  2.06433788e-01 ... -6.59191251e-01,  8.57466936e-01,  6.56416774e-01]]), None)
```

### 基于Trainer的快速训练、评测、推理

```python
import mindspore
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
# 初始化任务
vit_trainer = Trainer(
    task='image_classification',
    model='vit_base_p16',
    train_dataset="imageNet-1k/train",
    eval_dataset="imageNet-1k/val")
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1：使用现有的预训练权重进行finetune， 并使用finetune获得的权重进行eval和推理
vit_trainer.train(resume_or_finetune_from_checkpoint="mae_vit_base_p16", do_finetune=True)
vit_trainer.evaluate(eval_checkpoint=True)
predict_result = vit_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式2: 从新开始训练，并使用训练好的权重进行eval和推理
vit_trainer.train()
vit_trainer.evaluate(eval_checkpoint=True)
predict_result = vit_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式3： 从obs下载训练好的权重并进行eval和推理
vit_trainer.evaluate()
predict_result = vit_trainer.predict(input_data=img, top_k=3)
print(predict_result)

# output
# - mindformers - INFO - output result is: [[{'score': 0.8880876, 'label': 'daisy'},
# {'score': 0.0049882396, 'label': 'bee'}, {'score': 0.0031068476, 'label': 'vase'}]]
```

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
pipeline_task = pipeline("image_classification", model='vit_base_p16')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img, top_k=3)
print(pipeline_result)

# output
# [[{'score': 0.8880876, 'label': 'daisy'}, {'score': 0.0049882396, 'label': 'bee'},
# {'score': 0.0031068476, 'label': 'vase'}]]
```

 Trainer和pipeline接口默认支持的task和model关键入参

|    task（string）    | model（string） |
| :------------------: | :-------------: |
| image_classification |  vit_base_p16   |

## 预训练

### 数据集准备-预训练

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB

 ```text
数据集目录格式
└─imageNet-1k
    ├─train                # 训练数据集
    └─val                  # 评估数据集
 ```

### 脚本启动

#### 单卡训练

- python启动

```bash
# pretrain
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode train
```

#### 多卡训练

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/vit/run_vit_base_p16_224_100ep.yaml [0,8] train 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

**注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE ../configs/vit/run_vit_base_p16_224_100ep.yaml [0,8] train $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE ../configs/vit/run_vit_base_p16_224_100ep.yaml [$rank_start,$rank_end] train $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 评测

### 图像分类

### 数据集准备-图像分类

参考[数据集准备-预训练](#数据集准备-预训练)

### 脚本启动

#### 单卡评测

```bash
# evaluate
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode eval --eval_dataset_dir [DATASET_PATH]
# output
# ViT： Top1 Accuracy = {'Top1 Accuracy': 0.8371678937259923}
```

## 推理

### 脚本启动

#### 单卡推理

```bash
# predict
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode predict --predict_data [PATH_TO_IMAGE]
```
