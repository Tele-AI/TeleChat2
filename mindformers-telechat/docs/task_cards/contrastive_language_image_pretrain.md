# Contrastive Language Image Pretrain

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 任务描述

语言图像对比预训练：对模型进行图文对比学习，增强模型对文本图片的匹配度认识能力，预训练完的模型可用于零样本图像分类等下游任务

[相关论文](https://arxiv.org/abs/2103.00020) Alec Radford, Jong Wook Kim, et al., Learning Transferable Visual Models From Natural Language Supervision, 2021.

## 已支持数据集性能

| model |                                     type                                     | Datasets | Performance |  stage   |         example         |
|:-----:|:----------------------------------------------------------------------------:|:--------:|:-----------:|:--------:|:-----------------------:|
| clip  | clip_vit_b_32 <br> clip_vit_b_16 <br> clip_vit_l_14<br> clip_vit_l_14@336 | Flickr8k |     --      | pretrain | [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/contrastive_language_image_pretrain/clip_vit_b_32_pretrain_on_flickr8k.sh) |

### Flickr8k([链接](https://pan.baidu.com/s/1LRlQUL1MRipPL4MLOdExzg) 提取码: s4be)

- 数据集大小：2.2G，共8000张彩色图像，每张图像都与五个不同的标题配对，这些标题提供了对图片中物体和事件的内容描述
    - 训练集：6000张图像
    - 验证集：1000张图像
    - 测试集：1000张图像
- 数据格式：RGB

 ```bash
数据集目录格式
└─Flickr8k
    ├─Flickr8k_Dataset
    |      └─Flickr8k_Dataset
    └─Flickr8k_text
           ├─Flickr8k.devImages.txt
           ├─Flickr8k.testImages.txt
           ├─Flickr8k.trainImages.txt
           └─Flickr8k.token.txt
 ```

## 快速任务接口

- Trainer接口开启训练：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import MindFormerBook
from mindformers.trainer import Trainer

# 显示Trainer的模型支持列表
MindFormerBook.show_trainer_support_model_list("contrastive_language_image_pretrain")
# INFO - Trainer support model list for contrastive_language_image_pretrain task is:
# INFO -    ['clip_vit_b_32', 'clip_vit_b_16', 'clip_vit_l_14', 'clip_vit_l_14@336']
# INFO - -------------------------------------

# 初始化trainer
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='clip_vit_b_32',
    train_dataset='./Flickr8k'
)

trainer.train()
```
