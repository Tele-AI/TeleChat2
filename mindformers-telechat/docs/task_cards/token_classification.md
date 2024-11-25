# Token Classification

## 任务描述

命名实体识别：模型在基于命名实体识别数据集的微调后，可以在给定任意文本与候选标签列表的情况下，完成对文本中命名实体的识别。

[相关论文](https://arxiv.org/abs/2001.04351) Xu, Liang and Dong, Qianqian and Yu, Cong and Tian, Yin and Liu, Weitang and Li, Lu and Zhang, Xuanwei, CLUENER2020: Fine-grained Name Entity Recognition for Chinese, 2020.

## 已支持数据集性能

| model  |               type               | datasets | Entity F1 |           stage            |                           example                            |
| :----: | :------------------------------: | :------: | :-------: | :------------------------: | :----------------------------------------------------------: |
| tokcls | tokcls_bert_case_chinese_cluener | CLUENER  |  0.7905   | finetune<br>eval<br>predict | [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/token_classification/tokcls_bert_base_chinese_finetune_on_cluener.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/token_classification/tokcls_bert_base_chinese_eval_on_cluener.sh) <br> [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/token_classification/tokcls_bert_base_chinese_predict_on_cluener.sh) |

### [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)

- 数据集：训练集大小为10748个样本，验证集大小为1343个样本，10个类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）。
- 数据格式：json文件

 ```bash
数据集目录格式
└─cluener
    ├─train.json
    ├─dev.json
    ├─test.json
    ├─cluener_predict.json
 ```

## 快速任务接口

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

- 在脚本执行目录创建 `cluener` 文件夹，然后将数据集放入其中

- 脚本运行测试

```shell
# finetune
python run_mindformer.py --config ./configs/tokcls/run_tokcls_bert_base_chinese.yaml --run_mode finetune --load_checkpoint tokcls_bert_base_chinese

# evaluate
python run_mindformer.py --config ./configs/tokcls/run_tokcls_bert_base_chinese.yaml --run_mode eval --load_checkpoint tokcls_bert_base_chinese_cluener

# predict
python run_mindformer.py --config ./configs/tokcls/run_tokcls_bert_base_chinese.yaml --run_mode predict --load_checkpoint tokcls_bert_base_chinese_cluener --predict_data [TEXT]
```

### 调用API启动

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer

# 初始化trainer
trainer = Trainer(task='token_classification',
                  model='tokcls_bert_base_chinese',
                  train_dataset='./cluener/',
                  eval_dataset='./cluener/')
# 测试数据
input_data = ["结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"]

#方式1：使用现有的预训练权重进行finetune， 并使用finetune获得的权重进行eval和推理
trainer.finetune(finetune_checkpoint="tokcls_bert_base_chinese")
trainer.evaluate(eval_checkpoint=True)
trainer.predict(predict_checkpoint=True, input_data=input_data)

# 方式2： 从obs下载训练好的权重并进行eval和推理
trainer.evaluate()
# INFO - Entity F1=0.7853
trainer.predict(input_data=input_data)
# INFO - output result is [[{'entity_group': 'organization', 'start': 20, 'end': 24, 'score': 0.94914, 'word': '瓦拉多利德'},
#                           {'entity_group': 'organization', 'start': 33, 'end': 34, 'score': 0.9496, 'word': '西甲'}]]
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import TokenClassificationPipeline
from mindformers import AutoTokenizer, BertForTokenClassification, AutoConfig
from mindformers.dataset.labels import cluener_labels

input_data = ["表身刻有代表日内瓦钟表匠freresoltramare的“fo”字样。"]

id2label = {label_id: label for label_id, label in enumerate(cluener_labels)}

tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
tokcls_cluener_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener')

# This is a known issue, you need to specify batch size equal to 1 when creating bert model.
tokcls_cluener_config.batch_size = 1

model = BertForTokenClassification(tokcls_cluener_config)
tokcls_pipeline = TokenClassificationPipeline(task='token_classification',
                                              model=model,
                                              id2label=id2label,
                                              tokenizer=tokenizer,
                                              max_length=model.config.seq_length,
                                              padding="max_length")

results = tokcls_pipeline(input_data)
print(results)
# 输出
# [[{'entity_group': 'address', 'start': 6, 'end': 8, 'score': 0.52329, 'word': '日内瓦'},
#   {'entity_group': 'name', 'start': 12, 'end': 25, 'score': 0.83922, 'word': 'freresoltramar'}]]
```
