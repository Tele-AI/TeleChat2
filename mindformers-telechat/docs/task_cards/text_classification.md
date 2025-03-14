# Text Classification

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 任务描述

文本分类：模型在基于文本对的微调后，可以在给定任意文本对与候选标签列表的情况下，完成对文本对关系的分类，文本对的两个文本之间以-分割。

[相关论文](https://arxiv.org/pdf/1810.04805.pdf) Jacob Devlin, Ming-Wei Chang, et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019.

## 已支持数据集性能

| model  |                            type                            | datasets |  Top1-accuracy  |           stage            |                                                                                                                             example                                                                                                                              |
|:------:|:----------------------------------------------------------:|:--------:|:---------------:|:--------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  bert  | txtcls_bert_base_uncased |   Mnli   | 30.9% |          pretrain          |                                                                                                                                --                                                                                                                                |
| txtcls | txtcls_bert_case_uncased_mnli |   Mnli   | 84.8% | finetune<br>eval<br>predict | [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/text_classification/txtcls_bert_base_uncased_finetune_on_mnli.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/text_classification/txtcls_bert_base_uncased_mnli_eval_on_mnli.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/text_classification/txtcls_bert_base_uncased_mnli_predict_on_mnli.sh) |

### [Mnli](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip)

- 数据集大小：298M，共431992个样本，3个类别
    - 训练集：392702个样本
    - 匹配测试集：9796个样本
    - 非匹配测试集：9847个样本
    - 匹配开发集：9815个样本
    - 非匹配开发集：9832个样本
- 数据格式：tsv文件

 ```bash
数据集目录格式
└─mnli
    ├─dev
    ├─test  
    └─train
 ```

- 用户可以参考[BERT](https://github.com/google-research/bert)代码仓中的run_classifier.py文件，进行Mnli数据集`TFRecord`格式文件的生成。

## 快速任务接口

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

- 脚本运行测试

```shell
# finetune
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode finetune --load_checkpoint txtcls_bert_base_uncased

# evaluate
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode eval --load_checkpoint txtcls_bert_base_uncased_mnli

# predict
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode predict --load_checkpoint txtcls_bert_base_uncased_mnli --predict_data [TEXT]
```

### 调用API启动

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import MindFormerBook
from mindformers.trainer import Trainer

# 显示Trainer的模型支持列表
MindFormerBook.show_trainer_support_model_list("text_classification")
# INFO - Trainer support model list for txt_classification task is:
# INFO -    ['txtcls_bert_base_uncased']
# INFO - -------------------------------------

# 初始化trainer
trainer = Trainer(task='text_classification',
    model='txtcls_bert_base_uncased',
    train_dataset='./mnli/train',
    eval_dataset='./mnli/eval')
# 测试数据，该input_data有两个测试案例，即两个文本对，单个文本对的两个文本之间用-分割
input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
              "i don't know um do you do a lot of camping-I know exactly."]

#方式1：使用现有的预训练权重进行finetune， 并使用finetune获得的权重进行eval和推理
trainer.train(resume_or_finetune_from_checkpoint="txtcls_bert_base_uncased",
              do_finetune=True)
trainer.evaluate(eval_checkpoint=True)
trainer.predict(predict_checkpoint=True, input_data=input_data, top_k=1)

# 方式2： 从obs下载训练好的权重并进行eval和推理
trainer.evaluate()
# INFO - Top1 Accuracy=84.8%
trainer.predict(input_data=input_data, top_k=1)
# INFO - output result is [[{'label': 'neutral', 'score': 0.9714198708534241}],
#                         [{'label': 'contradiction', 'score': 0.9967639446258545}]]
```

- pipeline接口开启快速推理

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.pipeline import TextClassificationPipeline
from mindformers import AutoTokenizer, BertForMultipleChoice, AutoConfig

input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
                "i don't know um do you do a lot of camping-I know exactly."]

tokenizer = AutoTokenizer.from_pretrained('txtcls_bert_base_uncased_mnli')
txtcls_mnli_config = AutoConfig.from_pretrained('txtcls_bert_base_uncased_mnli')

# Because batch_size parameter is required when bert model is created, and pipeline
# function deals with samples one by one, the batch_size parameter is seted one.
txtcls_mnli_config.batch_size = 1

model = BertForMultipleChoice(txtcls_mnli_config)
txtcls_pipeline = TextClassificationPipeline(task='text_classification',
                                             model=model,
                                             tokenizer=tokenizer,
                                             max_length=model.config.seq_length,
                                             padding="max_length")

results = txtcls_pipeline(input_data, top_k=1)
print(results)
# 输出
# [[{'label': 'neutral', 'score': 0.9714198708534241}], [{'label': 'contradiction', 'score': 0.9967639446258545}]]
```
