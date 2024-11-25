# 接口支持度说明

## Datasets

| Dataset API                                                                                                                | 说明                                        |
|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| [mindformers.dataset.CausalLanguageModelDataset](mindformers.dataset.CausalLanguageModelDataset)                           | GPT类模型文本数据集，支持MindRecord，TFRecord，自定义数据集等 |
| [mindformers.dataset.ContrastiveLanguageImagePretrainDataset](mindformers.dataset.ContrastiveLanguageImagePretrainDataset) | CLIP类模型预训练图文对数据集，如Flickr8k等               |
| [mindformers.dataset.ImageCLSDataset](mindformers.dataset.ImageCLSDataset)                                                 | 图片分类数据集，如ImageNet2012等                    |
| [mindformers.dataset.KeyWordGenDataset](mindformers.dataset.KeyWordGenDataset)                                             | GLM模型文本数据集，支持MindRecord，自定义数据集等，如ADGen    |
| [mindformers.dataset.MaskLanguageModelDataset](mindformers.dataset.MaskLanguageModelDataset)                               | Bert类模型文本数据集，仅支持MindRecord，TFRecord       |
| [mindformers.dataset.MIMDataset](mindformers.dataset.MIMDataset)                                                           | Mae等图片数据集，支持MindRecord，ImageNet2012等      |
| [mindformers.dataset.TextClassificationDataset](mindformers.dataset.TextClassificationDataset)                             | 文本分类数据集，仅支持TFRecord                       |
| [mindformers.dataset.ZeroShotImageClassificationDataset](mindformers.dataset.ZeroShotImageClassificationDataset)           | 零样本分类图片数据集，支持自定义数据集，如Cifar100             |

## Learning Rate

| Learning Rate API                                                                                      | 说明                                                       |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [mindformers.core.lr.ConstantWarmUpLR](mindformers.core.lr.ConstantWarmUpLR)                           | LR从warmupLR开始线型增长到指定LR，warmup阶段后LR固定为LR×常数               |
| [mindformers.core.lr.CosineWithRestartsAndWarmUpLR](mindformers.core.lr.CosineWithRestartsAndWarmUpLR) | LR从warmupLR开始线型增长到指定LR，warmup阶段后进行cos(1)->cos(0) 衰减或周期变化 |
| [mindformers.core.lr.CosineWithWarmUpLR](mindformers.core.lr.CosineWithWarmUpLR)                       | LR从warmupLR开始线型增长到指定LR，warmup阶段后进行consine衰减或周期变化         |
| [mindformers.core.lr.LinearWithWarmUpLR](mindformers.core.lr.LinearWithWarmUpLR)                       | LR从warmupLR开始线型增长到指定LR，warmup阶段后LR从0开始增长到指定LR            |
| [mindformers.core.lr.PolynomialWithWarmUpLR](mindformers.core.lr.PolynomialWithWarmUpLR)               | LR从warmupLR开始线型增长到指定LR，warmup阶段后进行衰减                     |

## Loss

| Loss API                                                                                     | 说明                    |
|----------------------------------------------------------------------------------------------|-----------------------|
| [mindformers.core.loss.CrossEntropyLoss](mindformers.core.loss.CrossEntropyLoss)             | NLP常用的交叉熵损失函数         |
| [mindformers.core.loss.L1Loss](mindformers.core.loss.L1Loss)                                 | 平均绝对误差，SIMMIM所使用的损失函数 |
| [mindformers.core.loss.MSELoss](mindformers.core.loss.MSELoss)                               | 均方误差，Mae所使用的的损失函数     |
| [mindformers.core.loss.SoftTargetCrossEntropy](mindformers.core.loss.SoftTargetCrossEntropy) | Swin所使用的的损失函数         |

## Optimizer

| Optimizer API                                                                                      | 说明                                      |
|----------------------------------------------------------------------------------------------------|-----------------------------------------|
| [mindformers.core.optim.FusedAdamWeightDecay](mindformers.core.optim.FusedAdamWeightDecay)         | 权重衰减Adam算法的融合算子                         |
| [mindformers.core.optim.FP32StateAdamWeightDecay](mindformers.core.optim.FP32StateAdamWeightDecay) | 权重衰减Adam算法，与mindspore中一致，区别为优化器状态改为fp32 |
