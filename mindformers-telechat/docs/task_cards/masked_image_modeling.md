# Masked Image Modeling

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 任务描述

掩码图像建模：对图像中的部分图像块进行掩码，用剩下的图像块重建整张图像，从而对被掩码的图像块进行预测。

[相关论文-MAE](https://arxiv.org/abs/2111.06377): Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollár and Ross Girshick. 2021.

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

- Trainer接口开启训练/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import MindFormerBook
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 显示Trainer的模型支持列表
MindFormerBook.show_trainer_support_model_list("masked_image_modeling")
# INFO - Trainer support model list for masked_image_modeling task is:
# INFO -    ['mae_vit_base_p16']
# INFO - -------------------------------------

# 初始化trainer
mae_trainer = Trainer(
    task='masked_image_modeling',
    model='mae_vit_base_p16',
    train_dataset="imageNet-1k/train")
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1: 重头开始训练，并使用训练好的权重推理
mae_trainer.train()
predict_result = mae_trainer.predict(predict_checkpoint=True, input_data=img)
print(predict_result)

# 方式2： 从obs下载训练好的权重并进行eval和推理
predict_result = mae_trainer.predict(input_data=img)
print(predict_result)
# 输出
# [{'info': './output/output_image0.jpg', 'data': <PIL.Image.Image image mode=RGB size=224x224 at 0xFFFCFC2C0FD0>}]
```

- pipeline接口开启快速推理

```python
from mindformers import pipeline, MindFormerBook
from mindformers.tools.image_tools import load_image

# 显示pipeline支持的模型列表
MindFormerBook.show_pipeline_support_model_list("masked_image_modeling")
# INFO - Pipeline support model list for masked_image_modeling task is:
# INFO -    ['mae_vit_base_p16']
# INFO - -------------------------------------

# pipeline初始化
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image


pipeline_task = pipeline("masked_image_modeling", model='mae_vit_base_p16')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img)
print(pipeline_result)
# 输出
# [{'info': './output/output_image0.jpg', 'data': <PIL.Image.Image image mode=RGB size=224x224 at 0xFFFCFC2C0FD0>}]
```
