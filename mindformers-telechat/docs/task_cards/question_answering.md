# Question Answering

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 任务描述

**问答任务**：模型在基于问答数据集的微调后，输入为上下文（context）和问题（question），模型根据上下文（context）给出相应的回答。

**相关论文**

- Jacob Devlin, Ming-Wei Chang, et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf), 2019.
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang, [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf), 2016.

## 已支持数据集性能

| model |            type            |  datasets  |  EM   | F1    |           stage            |                           example                            |
| :---: | :------------------------: | :--------: | :---: | ----- | :------------------------: | :----------------------------------------------------------: |
|  q'a  | qa_bert_case_uncased_squad | SQuAD v1.1 | 80.74 | 88.33 | finetune<br>eval<br>predict | [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/question_answering/qa_bert_base_uncased_finetune_on_squad.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/question_answering/qa_bert_base_uncased_eval_on_squad.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/question_answering/qa_bert_base_uncased_predict_on_squad.sh) |

### [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)

- 下载地址：[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)，[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- 数据集：该数据集包含 10 万个（问题，原文，答案）三元组，原文来自于 536 篇维基百科文章，而问题和答案的构建主要是通过众包的方式，让标注人员提出最多 5 个基于文章内容的问题并提供正确答案，且答案出现在原文中。
- 数据格式：json文件

 ```bash
数据集目录格式
└─squad
    ├─train-v1.1.json
    └─dev-v1.1.json
 ```

## 快速任务接口

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

- 在脚本执行目录创建 `squad` 文件夹，然后将数据集放入其中

- 脚本运行测试

```shell
# finetune
python run_mindformer.py --config ./configs/qa/run_qa_bert_base_uncased.yaml --run_mode finetune --load_checkpoint qa_bert_base_uncased

# evaluate
python run_mindformer.py --config ./configs/qa/run_qa_bert_base_uncased.yaml --run_mode eval --load_checkpoint qa_bert_base_uncased_squad

# predict
python run_mindformer.py --config ./configs/qa/run_qa_bert_base_uncased.yaml --run_mode predict --load_checkpoint qa_bert_base_uncased_squad --predict_data [TEXT]
```

### 调用API启动

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer

# 初始化trainer
trainer = Trainer(task='question_answering',
                  model='qa_bert_base_uncased',
                  train_dataset='./squad/',
                  eval_dataset='./squad/')

#方式1：使用现有的预训练权重进行finetune， 并使用finetune获得的权重进行eval和推理
trainer.train(resume_or_finetune_from_checkpoint="qa_bert_base_uncased",
              do_finetune=True)
trainer.evaluate(eval_checkpoint=True)
# 测试数据，测试数据分为context和question两部分，两者以 “-” 分隔
input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]
trainer.predict(predict_checkpoint=True, input_data=input_data)

# 方式2： 从obs下载训练好的权重并进行eval和推理
trainer.evaluate()
# INFO - QA Metric = {'QA Metric': {'exact_match': 80.74739829706716, 'f1': 88.33552874684968}}
# 测试数据，测试数据分为context和question两部分，两者以 “-” 分隔
input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]
trainer.predict(input_data=input_data)
# INFO - output result is [{'text': 'Berlin', 'score': 0.9941, 'start': 34, 'end': 40}]
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import QuestionAnsweringPipeline
from mindformers import AutoTokenizer, BertForQuestionAnswering, AutoConfig

# 测试数据，测试数据分为context和question两部分，两者以 “-” 分隔
input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]

tokenizer = AutoTokenizer.from_pretrained('qa_bert_base_uncased_squad')
qa_squad_config = AutoConfig.from_pretrained('qa_bert_base_uncased_squad')

# This is a known issue, you need to specify batch size equal to 1 when creating bert model.
qa_squad_config.batch_size = 1

model = BertForQuestionAnswering(qa_squad_config)
qa_pipeline = QuestionAnsweringPipeline(task='question_answering',
                                        model=model,
                                        tokenizer=tokenizer)

results = qa_pipeline(input_data)
print(results)
# 输出
# [{'text': 'Berlin', 'score': 0.9941, 'start': 34, 'end': 40}]
```
