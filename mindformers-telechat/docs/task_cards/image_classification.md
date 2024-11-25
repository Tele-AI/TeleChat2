# Image Classification

## 任务描述

图像分类：模型基于图像数据集进行训练后，可以在给定任意图片的情况下，完成对图像的分类，分类结果仅限于数据集中所包含的类别。

[相关论文-vit](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F2010.11929): Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.
[相关论文-swin](https://arxiv.org/abs/2103.14030) Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo, 2021

## 已支持数据集性能

| model |      type      |  datasets   | Top1-accuracy |                stage                 |                                                                                                                                                                                        example                                                                                                                                                                                        |
| :---: | :------------: | :---------: | :-----------: | :----------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  vit  |  vit_base_p16  | ImageNet-1K |    83.71%     | train<br>finetune<br>eval<br>predict |     [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/vit_base_p16_train_on_imagenet1k.sh) <br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/vit_base_p16_finetune_on_imagenet1k.sh)<br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/vit_base_p16_eval_on_imagenet1k.sh)<br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/vit_base_p16_predict_on_imagenet1k.sh)     |
| swin  | swin_base_p4w7 | ImageNet-1K |    83.44%     | train<br>finetune<br>eval<br>predict | [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/swin_base_p4w7_train_on_imagenet1k.sh) <br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/swin_base_p4w7_finetune_on_imagenet1k.sh)<br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/swin_base_p4w7_eval_on_imagenet1k.sh)<br>[link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/image_classification/swin_base_p4w7_predict_on_imagenet1k.sh) |

### [ImageNet2012](http://www.image-net.org/)

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

## 快速任务接口

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import MindFormerBook
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 显示Trainer的模型支持列表
MindFormerBook.show_trainer_support_model_list("image_classification")
# INFO - Trainer support model list for image_classification task is:
# INFO -    ['vit_base_p16', 'swin_base_p4w7']
# INFO - -------------------------------------
# 下面以ViT模型为例，Swin同理

# 初始化trainer
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

# 方式2: 重头开始训练，并使用训练好的权重进行eval和推理
vit_trainer.train()
vit_trainer.evaluate(eval_checkpoint=True)
predict_result = vit_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式3： 从obs下载训练好的权重并进行eval和推理
vit_trainer.evaluate()
predict_result = vit_trainer.predict(input_data=img, top_k=3)
print(predict_result)
```

- pipeline接口开启快速推理

```python
from mindformers import pipeline, MindFormerBook
from mindformers.tools.image_tools import load_image

# 显示pipeline支持的模型列表
MindFormerBook.show_pipeline_support_model_list("image_classification")
# INFO - Pipeline support model list for image_classification task is:
# INFO -    ['vit_base_p16', 'swin_base_p4w7']
# INFO - -------------------------------------
# 下面以ViT模型为例，Swin同理

# pipeline初始化
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image


pipeline_task = pipeline("image_classification", model='vit_base_p16')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img, top_k=3)
# 输出
# [[{'score': 0.8846962, 'label': 'daisy'}, {'score': 0.005090589, 'label': 'bee'}, {'score': 0.0031510447, 'label': 'vase'}]]
```
