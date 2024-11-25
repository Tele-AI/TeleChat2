# GPT2从头开始实现

**注：本章节基于MindFormers提供Base_Trainer，Task等基础功能已实现的情况下，从头开始实现一个模型。**

## 1 模型构建及注册（以预训练语言模型为例）

### 1.1 熟悉模型架构

- 阅读GPT2论文[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，明确GPT2是Transformers Decoder结构。

![GPT2模型示例图](assets/gpt2.png)

- 查看[开源权威代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)，明确论文实现细节。确定`GPT2LMHeadModel`为模型预训练模块，需在`Mindformers`实现其及其依赖模块。

  ```txt
  Huggingface-GPT2：
    modeling_gpt2.py
        ├── GPT2Attention # Multi-head attention模块
          ├── GPT2MLP # transformer前馈层
          ├── GPT2Block # transformers block
          ├── GPT2Model # GPT2 backbone
          └── GPT2LMHeadModel # 模型预训练模块
  ```

- 分析需实现模块，明确`Mindformers`中的已有能力。上述`GPT2Block`是`transformer-decoder`的单层实现，`GPT2Attention`和`GPT2MLP`是`GPT2Block`的依赖模块。另`Mindformers`已集成整个`transformer`，故`GPT2Block`可基于`Mindformers`上的`transformer`实现。

  ```txt
  Mindformers
    ├── mindformers
        ├── modules
            ├── transformer
                └── transformer.py
  ```

- 基于`Mindformers`的`transformer`实现GPT2网络。

  最终GPT2网络的[整体目录](https://gitee.com/mindspore/mindformers/tree/dev/mindformers/models/gpt2)如下：

  ```txt
  Mindformers
    ├── mindformers
        ├── models
            ├── gpt2
                ├── __init__.py
                ├── convert_weight.py # 权重转化脚本
                ├── gpt2.py # 模型网络实现脚本
                ├── gpt2_config.py # 模型配置脚本
                ├── gpt2_processor.py # 预处理脚本
                ├── gpt2_tokenizer.py # tokenizer脚本
                └── gpt_modules.py # 模型transformer层实现脚本
  ```

  上述目录下详细代码参考

  [\_\_init\_\_.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/__init__.py)

  [convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/convert_weight.py)

  [gpt2.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2.py)

  [gpt2_config.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2_config.py)

  [gpt2_processor.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2_processor.py)

  [gpt2_tokenizer.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2_tokenizer.py)

  [gpt_modules.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt_modules.py)

- 实现GPT2 backbone网络`GPT2Model`（详细代码参考[gpt2.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2.py)）

  ```python
  # 具体实现参考Mindformers/mindformers/models/gpt2/gpt2.py
  class GPT2Model(nn.Cell):
      """ function description """
      def __init__(self, config):
          super(GPT2Model, self).__init__()

          self.embedding = GPTEmbeddingLayer(config) # embedding层
          self.layernorm = LayerNorm((config.hidden_size,)).to_float(config.layernorm_compute_type)

          self.blocks = nn.CellList()
          for i in range(config.num_layers):
              # GPTTransformerDecoderLayer为transformer block，如有需要，该部分可从transformer继承decoder重写
              block = GPTTransformerDecoderLayer(
                  hidden_size=config.hidden_size,
                  batch_size=config.batch_size,
                  ffn_hidden_size=config.hidden_size * config.expand_ratio,
                  seq_length=config.seq_length,
                  num_heads=config.num_heads,
                  attention_dropout_rate=config.attention_dropout_rate,
                  hidden_dropout_rate=config.hidden_dropout_rate,
                  hidden_act=config.hidden_act,
                  param_init_type=config.param_init_type,
                  layernorm_compute_type=config.layernorm_compute_type,
                  softmax_compute_type=config.softmax_compute_type,
                  parallel_config=config.parallel_config.dp_mp_config,
                  moe_config=moe_config)
              self.blocks.append(block)

          self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)

      def construct(self, input_ids, attention_mask):
          # word/position embedding
          input_embedding, embedding_table = self.embedding(input_ids, input_position)

          # multi-layer multi-head attention
          for i in range(self.num_layers):
              hidden_states = self.blocks[i](hidden_states, attention_mask)
          # layernorm
          output_state = self.layernorm(hidden_states)

          return output_state, embedding_table
  ```

- 基于`GPT2Model`实现GPT2预训练模块`GPT2LMHeadModel`（详细代码参考[gpt2.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2.py)）

     `__all__ = ['GPT2LMHeadModel']`：公开模型预训练接口，方便用户从外部调用。
     `@MindFormerRegister.register(MindFormerModuleType.MODELS)`：`Mindformers`的注册机制，将类注册到对应模块，方便通过配置文件进行实例化。
     `_support_list = MindFormerBook.get_model_support_list()['gpt2']`：高阶接口使用时用于检索可用模型，可通过`from_pretrained`的方法实例化。

  ```python
  # 具体实现参考Mindformers
  __all__ = ['GPT2LMHeadModel'] # 公开接口

  @MindFormerRegister.register(MindFormerModuleType.MODELS) # 注册到MODELS
  class GPT2LMHeadModel(PreTrainedModel):
      """ function description """
      _support_list = MindFormerBook.get_model_support_list()['gpt2']

      def __init__(self, config: GPT2Config = None):
          config = config if config is not None else GPT2Config()
          super(GPT2LMHeadModel, self).__init__(config, auto_prefix=True)

          # 用于生成下三角attention矩阵
          self.get_attention_mask = AttentionMask(seq_length=config.seq_length)

          self.backbone = GPT2Model(config) # gpt2 backbone
          # backbone网络输出隐向量向词表的映射矩阵
          self.head = GPTHead(hidden_size=config.hidden_size, vocab_size=config.vocab_size)
          # 交叉熵损失函数
          self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)

      def construct(self, input_ids, attention_mask=None):
          """ parameters description """
          # 生成下三角attention矩阵
          attention_mask = self.get_attention_mask(attention_mask)
          # 获取backbone网络的输出
          output_states, embedding_table = self.backbone(tokens, attention_mask)
          # backbone网络输出映射到词表空间
          logits = self.head(output_states, embedding_table)

          if self.phase != 'train':
              # 非训练时返回logits
              return logits, tokens, loss_mask
          # 训练时计算并返回loss
          loss = self.loss(logits, labels, loss_mask)
          return loss
  ```

- 编写模型参数配置文件`gpt2_config.py`（详细代码参考[gpt2_config.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/gpt2/gpt2_config.py)）
     `__all__ = ['GPT2Config']`：公开模型预训练接口，方便用户从外部调用。
     `@MindFormerRegister.register(MindFormerModuleType.CONFIG)`：`Mindformers`的注册机制，将类注册到对应模块，方便通过配置文件进行实例化。
     `_support_list = MindFormerBook.get_config_support_list()['gpt2']`：高阶接口使用时用于检索可用模型，可通过`from_pretrained`的方法实例化。

  ```python
  # 具体实现参考Mindformers/mindformers/models/gpt2/gpt2_config.py
  __all__ = ['GPT2Config']

  @MindFormerRegister.register(MindFormerModuleType.CONFIG) # 注册到CONFIG
  class GPT2Config(PretrainedConfig):
      """ class description """

      _support_list = MindFormerBook.get_config_support_list()['gpt2']

      def __init__(self,
                   seq_length: int = 1024,
                   vocab_size: int = 50257,
                   hidden_size: int = 768,
                   num_layers: int = 12,
                   num_heads: int = 12,
                   expand_ratio: int = 4,
                   embedding_dropout_prob: float = 0.1,
                   hidden_dropout_rate: float = 0.1,
                   attention_dropout_rate: float = 0.1,
                   param_init_type: str = "float32",
                   layernorm_compute_type: str = "float32",
                   softmax_compute_type: str = "float32",
                   compute_dtype: str = "float16",
                   hidden_act: str = 'gelu',
                   **kwargs):
          super(GPT2Config, self).__init__(**kwargs)
          self.seq_length = seq_length
          self.vocab_size = vocab_size
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          self.num_heads = num_heads
          self.expand_ratio = expand_ratio
          self.embedding_dropout_prob = embedding_dropout_prob
          self.hidden_dropout_rate = hidden_dropout_rate
          self.attention_dropout_rate = attention_dropout_rate
          self.param_init_type = convert_mstype(param_init_type)
          self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
          self.softmax_compute_type = convert_mstype(softmax_compute_type)
          self.compute_dtype = convert_mstype(compute_dtype)
          self.hidden_act = hidden_act
  ```

- 实现Trainer逻辑，支持`trainer.train/evaluate/predict`拉起`训练/评估/推理`流程。

  最终目录结构为：

  ``````txt
  Mindformers
    ├── mindformers
        ├── trainer
            ├── causal_language_modeling
                ├── __init__.py
                └── causal_language_modeling.py # train/evaluate/predict实现脚本
  ``````

  上述目录下详细代码参考：

  [\_\_init\_\_.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/causal_language_modeling/__init__.py)

  [causal_language_modeling.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/causal_language_modeling/causal_language_modeling.py)

     `@MindFormerRegister.register(MindFormerModuleType.TRAINER)`：`Mindformers`的注册机制，将类注册到对应模块，方便通过配置文件进行实例化
     `train/evaluate/predict`：`Mindformers`已实现`BaseTrainer`类，该类已实现基础训练功能，如训练流程有特殊需求可自行修改实现。

  ```python
  # 具体实现参考Mindformers/mindformers/trainer/causal_language_modeling/causal_language_modeling.py
  @MindFormerRegister.register(MindFormerModuleType.TRAINER)
  class CausalLanguageModelingTrainer(BaseTrainer):
      """ class description """
      def __init__(self, model_name: str = None):
          super(CausalLanguageModelingTrainer, self).__init__("text_generation", model_name)

      def train(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                wrapper: Optional[TrainOneStepCell] = None,
                optimizer: Optional[Optimizer] = None,
                callbacks: Optional[Union[Callback, List[Callback]]] = None,
                **kwargs):
          """ function description """
          self.training_process(
              config=config,
              network=network,
              callbacks=callbacks,
              dataset=dataset,
              wrapper=wrapper,
              optimizer=optimizer,
              **kwargs)

      def evaluate(self,
                   config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                   network: Optional[Union[Cell, PreTrainedModel]] = None,
                   dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                   callbacks: Optional[Union[Callback, List[Callback]]] = None,
                   compute_metrics: Optional[Union[dict, set]] = None,
                   **kwargs):
          """ function description """
          metric_name_list = [metric['type'] for metric in config.metric]
          if len(metric_name_list) == 1:
              metric_name = metric_name_list[0]
              kwargs.setdefault("metric_name", metric_name)

          self.evaluate_process(
                  config=config,
                  network=network,
                  dataset=dataset,
                  callbacks=callbacks,
                  compute_metrics=compute_metrics,
                  **kwargs)

      def predict(self,
                  config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                  input_data: Optional[Union[str, list, GeneratorDataset]] = None,
                  network: Optional[Union[Cell, PreTrainedModel]] = None,
                  tokenizer: Optional[PreTrainedTokenizerBase] = None,
                  **kwargs):
          """ function description """
          # 支持字符串和数据集传入
          if input_data is None:
              input_data = config.input_data

          if not isinstance(input_data, (str, list, GeneratorDataset)):
              raise ValueError("Input data's type must be one of "
                               f"[str, list, GeneratorDataset], but got type {type(input_data)}")

          if isinstance(input_data, str) and os.path.isfile(input_data):
              with open(input_data, 'r') as fp:
                  input_data_list = []
                  for line in fp:
                      input_data_list.extend(line)
              input_data = input_data_list

          return self.predict_process(config=config,
                                      input_data=input_data,
                                      task='text_generation',
                                      network=network,
                                      tokenizer=tokenizer,
                                      **kwargs)
  ```

- 配置yaml文件，在yaml配置文件中添加`训练/评估/推理`所需要的所有配置

  **需要区分不同参数规模的yaml文件**

  最终目录结构为：

  ``````txt
  Mindformers
      ├── configs
          ├── gpt2
              ├── run_gpt2.yaml # gpt2 small配置文件
              ├── run_gpt2_13b.yaml # gpt 13b配置文件
              ├── run_gpt2_52b.yaml # gpt 52b配置文件
              ├── run_gpt2_lora.yaml # gpt2 small lora微调配置文件
              ├── run_gpt2_txtcls.yaml # gpt2 small 文本分类配置文件
              ├── run_gpt2_xl.yaml # gpt2 xlarge配置文件
              └── run_gpt2_xl_lora.yaml # gpt2 xlarge lora微调配置文件
  ``````

  上述目录下详细代码参考

  [run_gpt2.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2.yaml)

  [run_gpt2_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_13b.yaml)

  [run_gpt2_52b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_52b.yaml)

  [run_gpt2_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_lora.yaml)

  [run_gpt2_txtcls.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_txtcls.yaml)

  [run_gpt2_xl.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_xl.yaml)

  [run_gpt2_xl_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/causal_language_modeling/causal_language_modeling.py)

  ```txt
  # 具体实现参考Mindformers
  run_gpt2.yaml # 配置文件名
      ├── context # 上下文配置
      ├── runner # batch_size和wrapper等配置
      ├── parallel # 分布式配置（网络中需适配分布式逻辑）
      ├── recompute_config # 重计算配置
      ├── profile # profile性能采集配置
      ├── trainer # trainer配置
        ├── type # 上述TRAINER注册，在此处可通过type进行实例化调用
        └── model_name # 模型名
      ├── train/eval_dataset # 训练/评估数据集配置
      ├── model # 模型参数配置
        ├── model # 模型参数配置
            ├── type # 上述CONFIG注册，在此处可通过type进行实例化调用
            ├── num_layers # transformers层数
            ├── num_heads # multi-head attention头数
            └── ......
        ├── arch
            └── GPT2LMHeadModel # 上述MODELS注册，在此处可通过该名称进行实例化调用
      ├── lr_schedule # 学习率配置（如使用自定义的，请先实现并注册，参考mindformers/core/lr）
      ├── optimizer # 优化器配置（如使用自定义的，请先实现并注册，参考mindformers/core/optim）
      ├── callbacks # 回调函数配置
      └── metric # 评估指标（如使用自定义的，请先实现并注册，参考mindformers/core/optim）
  ```

- `Mindformers/mindformer_book.py`中添加模型对应的检索名，可通过`from_pretrained`方法实例化（可选）

  ```python
  # 具体实现参考Mindformers/mindformers/mindformer_book.py
  _CONFIG_SUPPORT_LIST = OrderedDict([
          ('gpt2', [
              'gpt2',
          ]),
      ])

      _MODEL_SUPPORT_LIST = OrderedDict([
          ('gpt2', [
              'gpt2',
          ]),
      ])
  ```

  ```python
  # test from_pretrained
  from mindformers import GPT2Config, GPT2LMHeadModel

    print(GPT2Config._support_list)
    # ['gpt2']
    config = GPT2Config.from_pretrained("gpt2")

    print(GPT2LMHeadModel._support_list)
    # ['gpt2']
    model = GPT2LMHeadModel.from_pretrained("gpt2")
  ```

- 代码测试

  数据集构建和测试代码可参考[模型文档](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md)。

  **基于Mindformers完成模型实现后，建议撰写模型文档。**

## 2 数据集构建及注册

### 2.1 Processor

- `Processor`中初始化了`tokenizer`，对输入数据进行处理，`Mindformers`中已经实现了`ProcessorMixin`类，若无特别需求可直接继承该基类增加处理逻辑，如有需要可自行实现逻辑。
- `@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)`该装饰器对`GPT2Processor`进行了注册，将该类注册到了对应模块当中，由此可以通过配置文件进行实例化。
- `__all__ = ['GPT2Processor']`公开了模型预训练接口，方便用户从外部调用。
- `_support_list = MindFormerBook.get_config_support_list()['gpt2']`：高阶接口使用时用于检索可用模型，可通过`from_pretrained`的方法实例化`GPT2Processor`。

  ```python
  # gpt2_processor.py
  __all__ = ['GPT2Processor']

  @MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
  class GPT2Processor(ProcessorMixin):
    """class description"""
    _support_list = MindFormerBook.get_processor_support_list()['gpt2']

    def __init__(self, tokenizer=None,
                max_length=128, padding='max_length', return_tensors='ms'):
        super(GPT2Processor, self).__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors
        )

    def __call__(self, text_input=None, image_input=None):
        """call function description"""
        output = {}
        if text_input is not None and self.tokenizer:
            if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"tokenizer should inherited from the PreTrainedTokenizerBase,"
                                f" but got {type(self.tokenizer)}.")
            # 将输入数据增加batch维度
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input,return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            # 处理结果存入字典中
            output['text'] = text_output
        return output
    ```

### 2.2 Tokenizer

- `Tokenizer`通常用于对原始的文本数据进行转换，它将数据转为`token`，进而送入模型中进行处理。在`Mindformers`中已经实现了`Tokenizer`基类，`GPT2Tokenizer`可直接继承`Tokenizer`类进行具体逻辑的实现。如有需要可从头编写相关逻辑。
- `@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)`该装饰器对`GPT2Tokenizer`进行了注册，将该类注册到了对应模块当中，由此可以通过配置文件进行实例化。
- `__all__ = ['GPT2Tokenizer']`，公开模型预训练接口，方便用户从外部调用。
- `_support_list = MindFormerBook.get_config_support_list()['gpt2']`：高阶接口使用时用于检索可用模型，可通过`from_pretrained`的方法实例化`GPT2Tokenizer`。

    ```python
    # gpt2_tokenizer.py
    # 以下展示了部分核心代码，具体实现请参考Mindformers
    @MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
    class GPT2Tokenizer(PreTrainedTokenizer):
        """class description"""
        vocab_files_names = VOCAB_FILES_NAMES
        model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        FILE_LIST = ['tokenizer_config.json']
        _support_list = MindFormerBook.get_tokenizer_support_list()['gpt2']
        def __init__(self, vocab_file, merges_file,
                     errors="replace",
                     unk_token="<|endoftext|>",
                     bos_token="<|endoftext|>",
                     eos_token="<|endoftext|>",
                     pad_token="<|endoftext|>",
                     add_prefix_space=False,
                     add_bos_token=True,
                     add_eos_token=True,
                     **kwargs):
            # 调用基类AddedToken方法添加特殊token符
            bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
            eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
            unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
            pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
            super().__init__(errors=errors,unk_token=unk_token,bos_token=bos_token,eos_token=eos_token,
                             pad_token=pad_token,add_prefix_space=add_prefix_space,**kwargs)
            self.add_bos_token = add_bos_token
            self.add_eos_token = add_eos_token

            # 读取词表文件构造编码器与解码器
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.errors = errors  # how to handle errors in decoding
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

            # 读取合并文件构造BPE分词器
            with open(merges_file, encoding="utf-8") as merges_handle:
                bpe_merges = merges_handle.read().split("\n")[1:-1]
            bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
            self.cache = {}
            self.add_prefix_space = add_prefix_space

            self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        ...

        def bpe(self, token):
            """GPT2 BPE解码实现"""
            if token in self.cache:
                return self.cache[token]
            word = tuple(token)
            pairs = get_pairs(word)

            if not pairs:
                return token

            while True:
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                    except ValueError:
                        new_word.extend(word[i:])
                        break
                    else:
                        new_word.extend(word[i:j])
                        i = j

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)
            word = " ".join(word)
            self.cache[token] = word
            return word
    ```

### 2.3 Mindrecord文件生成

`MindRecord` 是 `MindSpore` 特有的一种数据格式，使得数据得以被高效地处理，输入数据转换为 `MindRecord` 的过程就是数据集处理的过程，该处理脚本需要用户自行编写。接下来以Wikitext2数据集为例，开发GPT2数据处理脚本。

- 数据集下载：[Wikitext2数据集](https://gitee.com/link?target=https%3A%2F%2Fs3.amazonaws.com%2Fresearch.metamind.io%2Fwikitext%2Fwikitext-2-v1.zip)

- 词表文件下载：[vocab.json](https://huggingface.co/gpt2/blob/main/vocab.json)，[merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt)

#### 依赖库导入

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import re
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers import AutoTokenizer
```

#### 数据读取

- 构造`preprocess_data`函数，传入`input_file`参数，根据相应路径逐行读入数据集文件内容。

```python

def preprocess_data(input_file):
  """ 数据读入 """
  dataset_valid = []
  passage = []
  count = 0
  with open(input_file, 'r', encoding='utf-8') as f:
      for line in f:
          line = line.strip()
          if line:
              if line.startswith('=') and line.endswith('=') and passage:
                  dataset_valid.append(passage)
                  count += 1
                  passage = []
              elif line.startswith('=') and line.endswith('='):
                  continue
              else:
                  passage.append(line)
  print('read {} file finished!\n total count = {}'.format(input_file, count))

  res = []
  for line in dataset_valid:
      text = ""
      for sentence in line:
          sentence = wikitext_clean(sentence)
          text = text + " " + sentence
      text = text.strip()
      res.append(text)
  return res
```

#### 数据清洗

- 根据Wikitext2数据集的格式特点编写数据清洗逻辑，构造`wikitext_clean`函数供`preprocess_data`方法调用。

```python
def wikitext_clean(string):
  """ 数据清洗 """
  # 消除空格
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # 处理数字分隔符
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # 处理标点符号
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" .", ".")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # 处理括号
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # 其他处理项
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")

  return string
```

#### 文本转换为token

- 实例化一个`GPT2Tokenizer`，编写`create_instance`函数，将`tokenizer`和相应的参数传入方法中，处理每条数据生成的`token`。

```python
def create_instance(tokenizer, sentence, ids, max_length=None):
  """文本转token"""
  pair_ids = None
  if len(sentence) == 2:
      pair_ids = tokenizer.encode(sentence[1])
  output = tokenizer.prepare_for_model(ids=ids,
                                          pair_ids=pair_ids,
                                          add_special_tokens=False,
                                          max_length=max_length,
                                          padding='max_length',
                                          truncate_direction="LEFT",
                                          return_overflowing_tokens=False,
                                          return_attention_mask=True)
  return output
```

#### Mindrecord格式转换

- 利用`Mindspore`构造的`writer`对象，将`instance`中每一条输入数据生成的三种数据`input_ids/attention_mask/labels`写入`mindrecord`中。

```python
def write_instance_to_file(writer, instance):
  """将数据写入mindrecord中"""
  input_ids = instance["input_ids"]
  attention_mask = instance["attention_mask"]
  labels = instance["input_ids"]

  features = collections.OrderedDict()
  features["input_ids"] = np.asarray(input_ids).astype(np.int32)
  features["attention_mask"] = np.asarray(attention_mask).astype(np.int32)
  features["labels"] = np.asarray(labels).astype(np.int32)

  # 转换为mindrecord
  writer.write_raw_data([features])

  return features
```

#### 主函数

- 最后完成主函数逻辑，控制整个数据处理流程。

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../wikitext-2/wiki.valid.tokens",
                        help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, default="../wikitext2_processed/wikitext-2.mindrecord",
                        help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help="The MindRecord file will be split into the number of partition. ")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length. ")
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        help="Tokenizer type, can be set to any tokenizer "
                            "if its relevant model supports prompt text classification. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "attention_mask"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")
    args = parser.parse_args()

    # 实例化GPT2Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([])

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", output_file)

    # 构造转换mindrecord的FileWriter类
    writer = FileWriter(output_file, args.num_splits)

    # 预定义mindrecord输出列：["input_ids", "attention_mask", "labels"]
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                "attention_mask": {"type": "int32", "shape": [-1]},
                "labels": {"type": "int32", "shape": [-1]}
                }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema, "lm-schema")

    # wikitext2数据预处理
    dataset_valid = preprocess_data(args.input_file)
    total_written = 0
    logging.info("***** Reading from  %s *****", input_file)
    text_total = "\n".join(dataset_valid)  # the logic of \n is copied from modelzoo
    sentence = text_total.strip().split("\t")
    block_size = args.max_length

    # 将文本转换为token
    total_ids = tokenizer.encode(sentence[0])
    total_length = len(total_ids)
    total_length = (total_length // block_size) * block_size
    print("total_length", total_length)

    # 数据分批转换为mindrecord
    for i in range(total_length // block_size):
        ids = total_ids[block_size*i:block_size*(i+1)]
        output = create_instance(tokenizer, sentence, ids, args.max_length)
        write_instance_to_file(writer, instance=output)
        total_written += 1

    # 将内存中的数据同步到磁盘，生成mindrecord文件
    writer.commit()
    logging.info("Wrote %d total instances", total_written)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

- 以上的代码进行合并后，便是处理wikitext2数据集的完整代码文件`mindformers/tool/dataset_preprocess/gpt2/wikitext2_data_process.py`，具体使用方法为：

    ```bash
    # 训练
    python mindformers/tools/dataset_preprocess/gpt2/wikitext2_data_process.py --input_file ./wikitext-2/wiki.train.tokens --output_file ./wikitext-2.train..mindrecord --max_length 1025
    # 评测
    python mindformers/tools/dataset_preprocess/gpt2/wikitext2_data_process.py --input_file ./wikitext-2/wiki.valid.tokens --output_file ./wikitext-2.valid.mindrecord --max_length 1024
    ```

### 2.4 Dataset构建

针对不同的下游任务，有诸多`dataset`的构建方法，接下来以`Causal Language Model Dataset`为例，为GPT2的训练构造数据集。

- `Mindformers`中实现了`BaseDataset`基类，用户可直接继承使用，如有特殊需求可自行实现。
- `@MindFormerRegister.register(MindFormerModuleType.DATASET)`该装饰器对`CausalLanguageModelDataset`进行了注册，将该类注册到了对应模块当中，由此可以通过配置文件进行实例化。
- 构造的`Dataset`类为`Mindspore`的通用数据集API，具体的[Dataset接口](https://mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.dataset.html)可见`Mindspore`官网说明。

    ```python
    # causal_language_model_dataset.py
    # 以下展示了部分核心代码，具体实现请参考Mindformers
    @MindFormerRegister.register(MindFormerModuleType.DATASET)
    class CausalLanguageModelDataset(BaseDataset):
        """ class description"""
        def __new__(cls, dataset_config: dict = None):
            logger.info("Now Create Causal Language Model Dataset.")
            # 考虑分布式训练/微调情况
            rank_id = get_real_rank()
            device_num = get_real_group_size()
            dataset_config = copy.deepcopy(dataset_config)
            cls.init_dataset_config(dataset_config)
            rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
            dataset_config.rank_id = rank_id
            dataset_config.device_num = device_num

            # 根据dataset_config构造dataset
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)

            type_cast_op = C.TypeCast(mstype.int32)
            if dataset_config.eos_reset:
                if cls._is_semi_full_batch() or cls._is_data_parallel():
                    rank_id = 0
                    dis = dataset_config.batch_size
                else:
                    # 每张卡都从完整的batch中获取其中的batch切片
                    dis = dataset_config.batch_size // device_num
                    if dataset_config.batch_size % device_num != 0:
                        raise ValueError(
                            f"batch size {dataset_config.batch_size} should be a multiple of device number {device_num}."
                            " You should change the args: per_batch_size.")

                # 定义数据集batch大小
                dataset = dataset.batch(dataset_config.batch_size,
                                        drop_remainder=dataset_config.drop_remainder,
                                        output_columns=dataset_config.input_columns)
                # 自定义数据增强操作函数
                map_func = lambda input_ids: get_input_data_batch_slice_map(input_ids,
                                                                            eos_token_id=dataset_config.eos_token_id,
                                                                            rank_id=rank_id,
                                                                            dis=dis)
                # 对数据集应用自定义数据增强
                if is_version_ge(mindspore.__version__, '1.11.0'):
                    dataset = dataset.map(operations=map_func, input_columns=dataset_config.input_columns,
                                        output_columns=dataset_config.output_columns)
                else:
                    dataset = dataset.map(operations=map_func, input_columns=dataset_config.input_columns,
                                        output_columns=dataset_config.output_columns,
                                        column_order=dataset_config.output_columns)
                # 对数据集的指定列名顺序进行排列
                dataset = dataset.project(columns=dataset_config.output_columns)

                for input_arg in dataset_config.output_columns:
                    dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
            else:
                dataset = dataset.batch(dataset_config.batch_size,
                                        drop_remainder=dataset_config.drop_remainder,
                                        output_columns=dataset_config.input_columns,
                                        num_parallel_workers=dataset_config.num_parallel_workers)
                dataset = dataset.project(columns=dataset_config.input_columns)
                for input_arg in dataset_config.input_columns:
                    dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)

            # 数据集重复的次数
            dataset = dataset.repeat(dataset_config.repeat)

            return dataset
    ```

## 3 Task构建及注册

**使用MindFormers已支持的task，请参考[Trainer API](Using_Api.md#trainer)，[Pipeline API](Using_Api.md#pipeline)，[Task](../task_cards/index.html)。**

GPT2作为大语言模型，其主要的task是文本生成和对话问答方面的内容，本小节将以文本生成任务为例，介绍实现GPT2文本生成的具体逻辑以及任务注册的过程。在`Mindformers`中，各类下游任务，如`文本生成`、`图像分类`、`对话问答`等任务以`Pipeline`的形式构建，目前`Mindformers`仓中已实现了`BasePipeline`基类。需要完成的`TextGenerationPipeline`可直接继承`BasePipeline`实现具体的逻辑，如需实现的下游任务类型较为复杂，则用户可自行从头编写`Pipeline`类的逻辑。

- `@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="text_generation")`该装饰器完成了对`TextGenerationPipeline`的注册，由此可以通过配置文件使用高级接口进行实例化。
- `TextGenerationPipeline`中包含了输入数据的前、中、后处理，主要涵盖以下的过程：
    1. 前处理：初始化`GPT2Tokenizer`，将输入`inputs`转换为`token`，即输出`input_ids`。
    2. 模型处理：获取前处理中的`input_ids`作为`model_inputs`，调用网络的`generate`方法完成网络前向过程的计算，得到输出文本的`output_ids`。
    3. 后处理：利用`tokenizer`将`output_ids`解码，把`token`重新转换为文字，得到最后的文本输出。

    ```python
    # text_generation_pipeline.py
    # 以下展示了部分核心代码，具体实现请参考Mindformers
    __all__ = ['TextGenerationPipeline']

    @MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="text_generation")
    class TextGenerationPipeline(BasePipeline):
        """class description"""
        _support_list = _setup_support_list(["gpt2", "glm"])
        return_name = 'text_generation'

        def __init__(self, model: Union[str, PreTrainedModel, Model],
                    tokenizer: Optional[PreTrainedTokenizerBase] = None,
                    **kwargs):
            # model/tokenizer输入类型判断
            if isinstance(model, str):
                if model in self._support_list or os.path.isdir(model):
                    if tokenizer is None:
                        tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                    model = AutoModel.from_pretrained(model)
                    if not isinstance(tokenizer, PreTrainedTokenizerBase):
                        raise TypeError(f"tokenizer should be inherited from"
                                        f" PreTrainedTokenizerBase, but got {type(tokenizer)}.")
                else:
                    raise ValueError(f"{model} is not supported by {self.__class__.__name__},"
                                    f"please selected from {self._support_list}.")
            if not isinstance(model, (PreTrainedModel, Model)):
                raise TypeError(f"model should be inherited from PreTrainedModel or Model, but got type {type(model)}.")
            if tokenizer is None:
                raise ValueError(f"{self.__class__.__name__}"
                                " requires for a tokenizer.")
            super().__init__(model, tokenizer, **kwargs)

        # Pipeline输入参数处理
        def _sanitize_parameters(self, **pipeline_parameters):
            """function description"""
            # 数据前处理参数
            preprocess_keys = ['keys', 'add_special_tokens']
            preprocess_params = {}
            for item in preprocess_keys:
                if item in pipeline_parameters:
                    preprocess_params[item] = pipeline_parameters.get(item)

            # 后处理参数
            postprocess_params = {}

            # 模型处理参数
            forward_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length', 'seed']
            forward_kwargs = {}
            for item in forward_key_name:
                if item in pipeline_parameters:
                    forward_kwargs[item] = pipeline_parameters.get(item)
            return preprocess_params, forward_kwargs, postprocess_params

        # Pipeline预处理
        def preprocess(self, inputs: Union[str, dict, Tensor],
                    **preprocess_params):
            """function description"""
            add_special_tokens = preprocess_params.get('add_special_tokens', True)

            # 输入数据获取
            if isinstance(inputs, dict):
                keys = preprocess_params.get('keys', None)
                default_src_language_name = 'text'
                feature_name = keys.get('src_language', default_src_language_name) if keys else default_src_language_name
                inputs = inputs[feature_name]
                if isinstance(inputs, mindspore.Tensor):
                    inputs = inputs.asnumpy().tolist()

            # 利用tokenizer将文本转换为token
            input_ids = self.tokenizer(inputs, return_tensors=None, add_special_tokens=add_special_tokens)["input_ids"]
            return {"input_ids": input_ids}

        # Pipeline模型处理
        def forward(self, model_inputs: dict,
                    **forward_params):
            """function description"""
            forward_params.pop("None", None)
            input_ids = model_inputs["input_ids"]

            # 输入token到网络中进行计算，并生成相应文本的token序列
            output_ids = self.network.generate(input_ids, **forward_params)
            return {"output_ids": output_ids}

        # Pipeline后处理
        def postprocess(self, model_outputs: dict,
                        **postprocess_params):
            """function description"""
            # 输出token序列重新解码为文本
            outputs = self.tokenizer.decode(model_outputs["output_ids"], skip_special_tokens=True)
            return [{self.return_name + '_text': outputs}]
    ```

- 在`text_generation_pipeline.py`中调用了`self.model`的`generate`方法，`Mindformers`中已经在`mindfirmers/models/text_generator.py`实现了该方法的具体逻辑，用户可直接使用，如有特殊要求则可参考该脚本自行实现文本生成任务，以下展示部分核心逻辑：

    ```python
    # text_generator.py
    # 以下展示了部分核心代码，具体实现请参考Mindformers
    class GenerationMixin:
        """class description"""
        def __init__(self):
            pass  
        ...

        def _forward(self, origin_inputs, top_k, top_p, repetition_penalty, max_length, eos_token_id, streamer=None, pad_token_id=None):
            """function description"""
            if pad_token_id is None:
                pad_token_id = 0
            use_pynative = True

            # 如传入steamer，则使用流式推理功能
            if streamer is not None:
                streamer.put(origin_inputs[0])

            # 获取输入数据shape与模型参数设置
            batch_size = origin_inputs.shape[0]
            is_encoder_decoder = self.config.is_encoder_decoder
            logger.debug("The input shape is: %s", origin_inputs.shape)
            valid_length_each_example = []
            for i in range(batch_size):
                # 得到输入token序列真实长度(排除special token)
                valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != pad_token_id)) + 1)
            valid_length_each_example = np.array(valid_length_each_example)
            logger.debug("Get the valid for each example is: %s", valid_length_each_example)
            if not is_encoder_decoder and np.max(valid_length_each_example) > max_length:
                raise ValueError("The max_length set is smaller than the length in the input_ids. You shout set "
                                f"max_length to {np.max(valid_length_each_example)}")
            target_length = self.config.seq_length if max_length > self.config.seq_length else max_length
            logger.debug("max target_length is: %s", target_length)
            frequency_list = None
            # 将输入token序列pad至target_length
            input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs, pad_token_id=pad_token_id)
            logger.debug("pad the origin inputs from %s into shape: %s", origin_inputs.shape, input_ids.shape)

            # 生成input_mask
            input_mask = np.zeros_like(input_ids)
            for i in range(valid_length_each_example.shape[0]):
                input_mask[i, :valid_length_each_example[i]] = 1
            encoder_output = None
            encoder_mask = None

            if is_encoder_decoder:
                ...

            # 循环生成token直到is_finished中全部为True
            is_finished = [False] * batch_size

            # 若config.use_past为True则使用增量推理
            if self.config.use_past:
                self.is_first_iteration = True
            is_first_iteration = False

            #开始生成文本
            while np.sum(is_finished) != batch_size:
                if is_encoder_decoder:
                    ...
                else:
                    seq_length = input_ids.shape[1]
                    current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
                    logger.debug("validate length: %s", valid_length_each_example)

                    # 使用增量推理
                    if self.config.use_past:
                        is_first_iteration = self.is_first_iteration

                        # 生成推理所需的position_ids和attention_mask
                        position_ids, attention_mask = self.generate_pos_id_and_mask_for_incr_infer(
                            input_ids=input_ids,
                            current_index=current_index,
                            valid_length_each_example=valid_length_each_example
                        )
                        # 推理
                        logits = self._incremental_infer(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            current_index=current_index,
                            valid_length_each_example=valid_length_each_example
                        )[0]
                    # 不使用增量推理
                    else:
                        logits = self(Tensor(input_ids, mstype.int32))[0]  # pylint: disable=E1102
                    logits = logits.reshape(-1, logits.shape[-1])
                    log_probs = self.process_logits(logits, Tensor(current_index, mstype.int32),
                                                    is_first_iteration, self.config.use_past)

                # 对模型输出logits进行采样，生成文本token的选取概率p与索引p_args
                log_probs = log_probs.asnumpy()
                vocab_size = log_probs.shape[-1]
                if repetition_penalty != 1 and frequency_list is None:
                    frequency_list = np.array([[0 for _ in range(vocab_size)]])
                log_probs_revised = log_probs.reshape(batch_size, vocab_size)
                if repetition_penalty != 1:
                    log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                        (frequency_list > 0) * repetition_penalty
                p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)

                # 根据p与p_args随机选取生成文本
                for i in range(batch_size):
                    if is_finished[i]:
                        continue

                    # 根据p的概率大小对p_args进行采样
                    target_index = np.random.choice(len(p[i]), p=p[i])
                    target = p_args[i][target_index]

                    if repetition_penalty != 1:
                        frequency_list[0][target] = frequency_list[0][target] + 1
                    input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                    # 流式推理
                    if streamer is not None:
                        streamer.put(np.asarray([target]))

                    if is_encoder_decoder:
                        target_mask[i][valid_length_each_example[i]] = int(1)

                    valid_length_each_example[i] += int(1)
                    input_mask[i][valid_length_each_example[i] - 1] = 1

                    # 是否达到生成停止条件
                    if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length:
                        is_finished[i] = True
                        continue

            # 文本全部生成完毕，返回所有output_ids
            output_ids = []
            for i in range(batch_size):
                output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
            logger.debug("The output is: %s", output_ids)
            if streamer is not None:
                streamer.end()
            return output_ids
    ```

## 4 GPT2训练与推理实现

### 4.1 预训练与微调

- 实现GPT2的训练与微调脚本，可通过`Mindformers`中`scripts/run_distribute.sh`脚本拉起预训练或微调。训练逻辑目前已在`mindformers/trainer/base_trainer.py`实现，用户可直接使用该基类，如对训练流程有特殊需求则可修改`training_process`方法自定义训练流程。

    ```python
    # base_trainer.py
    # 以下展示了部分核心代码，具体实现请参考Mindformers
    def training_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            optimizer: Optional[Optimizer] = None,
            wrapper: Optional[TrainOneStepCell] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            **kwargs):
        """训练及微调逻辑实现"""

        # 获取训练/微调参数设置
        self.kwargs = kwargs
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # 若resume_training为True，且load_checkpoint为正确的ckpt路径，则预训练ckpt进行读取
        if config.resume_training and config.load_checkpoint:
            logger.info(".............Start load resume context from checkpoint..................")
            load_resume_context_from_checkpoint(config)

        # 构建数据集
        logger.info(".........Build Dataset For Train..........")
        if dataset is None:
            dataset = self.create_train_dataset()
        self.set_train_dataset(dataset)
        check_runner_config(config, dataset)
        if config.runner_config.sink_mode:
            epoch_num = math.ceil((config.runner_config.epochs - config.runner_config.initial_epoch)
                                  * config.runner_config.sink_size / dataset.get_dataset_size())
            dataset._dataset_helper = DatasetHelper(dataset, config.runner_config.sink_mode,
                                                    config.runner_config.sink_size, epoch_num)

        # 构建训练网络
        logger.info(".........Build Net For Train..........")
        if network is None and wrapper is None and \
                self.model_wrapper is None and self.network is None:
            if self.get_pipeline_stages() > 1:
                # 如果采用流水线并行，则采用流水线网络的构建方法
                network = self.create_pipeline_network(default_args={"parallel_config": config.parallel_config,
                                                                     "moe_config": config.moe_config})

            else:
                # 不采用流水线并行则采用普通构建的方法
                network = self.create_network(default_args={"parallel_config": config.parallel_config,
                                                            "moe_config": config.moe_config})
        # 如果network与wrapper都为空，则创建一个网络
        elif network is None and wrapper is None and self.network is not None:
            logger.info(".........Using The Existing Network For Train:: %s", self.network.__class__.__name__)
            network = self.network

        if network is not None:
            self.set_network(network, is_train=True)

        if wrapper is not None:
            self.set_model_wrapper(wrapper)

        self.count_parameters()

        # 构建优化器
        logger.info(".........Build Optimizer For Train..........")
        if optimizer is None and wrapper is None and self.model_wrapper is None:
            optimizer = self.create_optimizer_scheduler(network, layer_scale=config.layer_scale)

        # 构建模型wrapper
        if wrapper is None and self.model_wrapper is None:
            logger.info(".........Build Running Wrapper From Config For Train..........")
            wrapper = self.create_model_wrapper(network, optimizer)
        elif wrapper is None and self.model_wrapper is not None:
            logger.info(".........Using The Existing Model Wrapper: %s", self.model_wrapper.__class__.__name__)
            wrapper = self.model_wrapper

        # 构建模型callback
        logger.info(".........Build Callbacks For Train..........")
        if callbacks is None:
            callbacks = self.create_callbacks(default_args={
                "learning_rate": optimizer.learning_rate if optimizer else wrapper.optimizer.learning_rate,
                "origin_epochs": config.runner_config.origin_epochs,
                "dataset_size": config.data_size,
                "micro_batch_interleave_num": config.micro_batch_interleave_num,
                "micro_batch_num": config.parallel_config.micro_batch_num,
                "initial_epoch": config.runner_config.initial_epoch})

        # 定义在训练中评估的计算指标
        compute_metrics = None
        if config.do_eval:
            compute_metrics = self.create_metrics()

        # 利用Model高阶接口定义具体模型
        logger.info(".........Starting Init Train Model..........")
        if wrapper is not None:
            model = Model(wrapper, metrics=compute_metrics, eval_network=network)
        else:
            model = Model(network, optimizer=optimizer, metrics=compute_metrics, eval_network=network)

        # 训练恢复/微调流程
        if config.load_checkpoint or config.only_save_strategy:
            if config.resume_training:
                logger.info(".............Start resume training from checkpoint..................")
                transform_and_load_checkpoint(config, model, network, dataset, optimizer=optimizer)
            else:
                if config.load_checkpoint in SUPPORT_MODEL_NAMES:
                    config.load_checkpoint = \
                        AutoModel.from_pretrained(config.load_checkpoint).default_checkpoint_download_path
                transform_and_load_checkpoint(config, model, network, dataset)

        # 边训练边评估特性
        if config.do_eval:
            logger.info(".........Build Evaluate in Training Callback..........")
            eval_dataset = kwargs.get('eval_dataset', None)
            if eval_dataset is None:
                eval_dataset = self.create_eval_dataset()

            eval_callback = EvalCallBack(
                partial(
                    self._evaluate_in_training,
                    model=model,
                    eval_dataset=eval_dataset,
                ),
                step_interval=config.eval_step_interval if config.eval_step_interval else -1,
                epoch_interval=config.eval_epoch_interval if config.eval_epoch_interval else 1,
            )
            callbacks.append(eval_callback)

        # 模型训练
        logger.info(".........Starting Training Model..........")
        logger.info(".........Model Compiling, Please Wait a Moment...........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.sink_size,
                    initial_epoch=config.runner_config.initial_epoch)
        logger.info(".........Training Over!.............")
    ```

### 4.2 脚本启动

- 完成训练逻辑的编写后，下一步可通过配置预训练或微调所需要的`config`文件，利用`Mindformers`准备好的训练脚本拉起任务，以`configs/gpt2/run_gpt2_13b.yaml`为例：

    ```yaml
    # pretrain_gpt2_13b_fp16.yaml
    # 以下展示了部分配置，具体配置内容请参考Mindformers
    seed: 0
    run_mode: 'train'
    output_dir: './output' # path to save checkpoint/strategy
    load_checkpoint: ""
    src_strategy_path_or_dir: ''
    auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
    only_save_strategy: False
    resume_training: False

    # context
    context:
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "Ascend"
    enable_graph_kernel: False
    max_call_depth: 10000
    max_device_memory: "30GB"
    save_graphs: False
    save_graphs_path: "./graph"
    device_id: 0

    ...
    ```

- 然后以单机八卡为例，配置好训练`config`后，拉起训练任务还需要准备`RANK_TABLE_FILE`的json文件，运行`mindformers/tools/hccl_tools.py`可生成`rank_table`。

    ```bash
    python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
    ```

- 最后可通过`scripts/run_distribute.sh`拉起分布式训练。`RANK_TABLE_FILE`为`rank_table`的路径；当`RUN_STATUS`为`train`时启动预训练流程，当输入`finetune`时为微调流程。

    ```bash
    cd scripts
    bash run_distribute.sh {RANK_TABLE_FILE} ../configs/gpt2/pretrain_gpt2_13b_fp16.yaml [0,8) {RUN_STATUS}
    ```
