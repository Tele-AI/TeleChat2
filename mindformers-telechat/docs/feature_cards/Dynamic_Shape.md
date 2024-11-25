# 静态图使用动态shape微调

## 功能

Mindspore在2.3.0版本之后为finetune提供了动态shape能力

当前版本的动态shape仅包含动态SeqLength。基本原理是尽量减小计算的shape。动态SeqLength不再将数据统一padding到SeqLength长度，而是统一padding到该batch内最长数据的长度，既不损失精度又减小了计算量。

finetune开启动态shape，最大的优点在于精度不变的情况下性能可以提高数倍。以通用的alpaca-en-52k数据集为例，静态shape下llama2 7b单机8卡跑完两个Epoch的finetune，最佳性能大约是4小时；启用动态shape后，这个时间可以缩短到37分钟。

## 使用方式

在yaml中开启动态shape配置并选择动态shape支持的data_loader。

参考配置

```bash
train_dataset: &train_dataset
  data_loader:
    type: SFTDataLoader
    dataset_dir: '/path/to/alpaca_data_en_52k.json'
    tokenizer:
      unk_token: '<unk>'
      bos_token: '<s>'
      eos_token: '</s>'
      pad_token: '<unk>'
      type: LlamaTokenizer
      vocab_file: '/path/to/tokenizer.model'
    max_length: 4097
    file_format: json
    dataset_name: multi-insruct-dyn
    shuffle: False
  divisor: 2
  remainder: 1
  input_columns: ['input_ids', 'labels']
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 2
  repeat: 1
  numa_enabel: False
  prefetch_size: 1
  dynamic_batch: True
```

### data_loader相关参数：

- type: data_loader类型，如果是SFT数据任务，需要使用SFTDataLoader，否则无法使用SFT_MAP_FUNCTIONS解析数据
- tokenizer: 同下方processor中的tokenizer。因为动态shape是直接读json进来转成input_ids，所以需要tokenizer能力
- dataset_name：data_loader类型，根据要解析的数据选择。例如读原始alpaca的json数据，推荐使用multi-instruct-dyn类型；读mindformers中alpaca_converter.py格式化过的json数据时(即转mindrecord前的json数据)，推荐使用multi-round-chat-dyn类型。
- divisor: 动态shape网络张量seqlen的约数([详细文档](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.Symbol.html))
- remainder: 动态shape网络张量seqlen的余数([详细文档](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.Symbol.html))
- dynamic_batch: 动态shape开关

### model_config相关参数：

- is_dynamic: 静态图的动态shape开关。需要一并开启，否则会导致CausalMask的shape不匹配

### 关于自定义prompt

prompt对齐，常用在需要和其它baseline对齐的场景。是否对齐以最终输入模型的input_ids、labels以及其它相关的mask是否和目标对齐了为准。
一般把没对齐的部分decode出来，并修改对应的prompt内容来完成。如果是padding或者mask字符不一致导致的误差，可以通过修改data_loader中的pad_token_id或者map_function_kwargs中ignore_token_id参数解决。

### prompt示例

这里分别给出llama2和llama3和原mindrecord数据对齐时候的prompt供参考

llama2：

```bash
train_dataset: &train_dataset
  data_loader:
    map_function_kwargs: {"user_prompt":"A chat between a curious urer and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", "user_prompt_role":"USER: ", "assistant_prompt_role":"ASSISTANT:"}
```

llama3:

```bash
train_dataset: &train_dataset
  data_loader:
    map_function_kwargs: {"user_prompt":"A chat between a curious urer and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", "user_prompt_role":"USER: ", "assistant_prompt_role":" ASSISTANT:", "bos_token":"", "sep_token":" "}
```
