mindformers.models.PreTrainedTokenizer
======================================

.. py:class:: mindformers.models.PreTrainedTokenizer(**kwargs)

    所有慢速分词器的基类。

    该类处理所有关于分词和特殊符号的共享方法。同时包括下载、缓存和加载预训练分词器的方法，以及向词汇表中添加词汇的方法。
    此外，它以统一的方式包含了所有分词器中的添加词汇，简化了各种底层词典结构（如BPE、sentencepiece等）的词汇增强方法的处理。

    .. note::
        初始化分词器的基本配置。

        步骤：

        1. 初始化父类。
        2. 如果子类没有初始化 `_added_tokens_decoder` ，则进行初始化。
        3. 使用传入的 `added_tokens_decoder` 更新 `_added_tokens_decoder` 。
        4. 将那些不在词汇表中的特殊词元添加到词汇表中，添加的顺序与 `tokenizers` 中的 `SPECIAL_TOKENS_ATTRIBUTES` 相同。

        特点：

        1. 确保所有特殊词元均被添加到词汇表中，即使它们最初不在词汇表中。
        2. 使用Trie结构来存储词元。

    参数：
        - **model_max_length** (int, 可选) - 转换器模型输入的最大长度（以词元数量计）。当通过 `from_pretrained()` 加载分词器时，此值将设置为 `max_model_input_sizes` 中存储的关联模型的值。默认值： ``1e-30`` 。
        - **padding_side** (str, 可选) - 模型应该在哪一侧应用填充。应从['right', 'left']中选择。默认值从同名类属性中选择。
        - **truncation_side** (str, 可选) - 模型应该在哪一侧应用截断。应从['right', 'left']中选择。默认值从同名类属性中选择。
        - **chat_template** (str, 可选) - 将用于格式化聊天消息列表的Jinja模板字符串。默认值： ``"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"`` 。
        - **model_input_names** (List[str], 可选) - 模型前向传递接受的输入列表（如"token_type_ids"或"attention_mask"）。默认值从同名类属性中选择。
        - **bos_token** (Union[str, tokenizers.AddedToken], 可选) - 表示句子开始的特殊词元。将关联到self.bos_token和self.bos_token_id。默认值： ``None`` 。
        - **eos_token** (Union[str, tokenizers.AddedToken], 可选) - 表示句子结束的特殊词元。将关联到self.eos_token和self.eos_token_id。默认值： ``None`` 。
        - **unk_token** (Union[str, tokenizers.AddedToken], 可选) - 表示词汇表外词元的特殊词元。将关联到self.unk_token和self.unk_token_id。默认值： ``None`` 。
        - **sep_token** (Union[str, tokenizers.AddedToken], 可选) - 在同一输入中分隔两个不同句子的特殊词元（例如BERT使用）。将关联到self.sep_token和self.sep_token_id。默认值： ``None`` 。
        - **pad_token** (Union[str, tokenizers.AddedToken], 可选) - 用于使词元数组大小相同，以便批处理的特殊词元。注意机制或损失计算将忽略它。将关联到self.pad_token和self.pad_token_id。默认值： ``None`` 。
        - **cls_token** (Union[str, tokenizers.AddedToken], 可选) - 表示输入类的特殊词元（例如BERT使用）。将关联到self.cls_token和self.cls_token_id。默认值： ``None`` 。
        - **mask_token** (Union[str, tokenizers.AddedToken], 可选) - 表示掩码词元的特殊词元（用于掩码语言建模预训练目标，如BERT）。将关联到self.mask_token和self.mask_token_id。默认值： ``None`` 。
        - **additional_special_tokens** (Union[tuple, list, tokenizers.AddedToken], 可选) - 一组额外的特殊词元。在这里添加它们以确保在设置skip_special_tokens为True时跳过它们。如果它们不是词汇表的一部分，将在词汇表的末尾添加。默认值： ``None`` 。
        - **clean_up_tokenization_spaces** (bool, 可选) - 是否清理在分词过程中添加的空格。默认值： ``True`` 。
        - **split_special_tokens** (bool, 可选) - 是否在分词过程中拆分特殊词元。传递将影响分词器的内部状态。默认行为是不拆分特殊词元。这意味着如果 `<s>` 是 `bos_token` ，则 ``tokenizer.tokenize("<s>") = ['<s>']`` 。否则，如果 ``split_special_tokens=True`` ，则 ``tokenizer.tokenize("<s>")`` 会得到 ``['<','s', '>']`` 。默认值： ``False`` 。

    返回：
        PreTrainedTokenizer类实例。

    .. py:method:: added_tokens_decoder()
        :classmethod:

        以索引到AddedToken的字典形式，返回词汇表中的添加词元。

        返回：
            dict，添加的词元。

    .. py:method:: added_tokens_encoder()
        :classmethod:

        返回字符串到索引的排序映射。为了性能优化，添加的词元编码器在 `self._added_tokens_encoder` 中被缓存。

        返回：
            dict，添加的词元。

    .. py:method:: convert_ids_to_tokens(ids: Union[int, List[int]], skip_special_tokens: bool = False)

        使用词汇表和添加的词元，将单个索引或索引序列转换为词元或词元序列。

        参数：
            - **ids** (Union[int, list[int]]) - 要转换为词元的词元索引或词元索引序列。
            - **skip_special_tokens** (bool, 可选) - 是否在解码时移除特殊词元。默认值： ``False`` 。

        返回：
            解码的词元或词元列表，类型为 `str` 或 `List[str]` 。

    .. py:method:: convert_tokens_to_ids(tokens: Union[str, List[str]])

        使用词汇表将词元字符串（或词元序列）转换为单个整数索引（或索引序列）。

        参数：
            - **tokens** (Union[str, List[str]]) - 要转换为词元索引的一个或多个词元。

        返回：
            `ids` ，类型为 `int` 或 `List[int]` 的词元索引或词元索引序列。

    .. py:method:: get_added_vocab()

        以词元到索引的字典形式返回词汇表中的添加词元。

        返回：
            dict: 添加的词元。

    .. py:method:: num_special_tokens_to_add(pair: bool = False)

        返回在编码序列时添加的特殊词元的数量。

        .. note::
            这将编码一个虚拟输入并检查添加的词元数量，因此效率不高。不要将此方法放在您的训练循环中。

        参数：
            - **pair** (bool, 可选) - 是否在序列对的情况下计算添加的词元数量。默认值： ``False`` 。

        返回：
            序列中添加的特殊词元的数量。

    .. py:method:: prepare_for_tokenization(text: str, **kwargs)

        在分词前进行必要的转换。

        参数：
            - **text** (str) - 要准备的文本。
            - **kwargs** (Any, 可选) - 用于标记化的关键字参数。

        返回：
            一个类型为 `Tuple[str, dict]` 的元组，表示准备好的文本和未使用的kwargs。

    .. py:method:: tokenize(text: TextInput, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs)

        将字符串转换为词元序列，使用分词器。

        按单词拆分基于单词的词汇，或按子单词拆分基于子单词的词汇（BPE/SentencePieces/WordPieces）。处理添加的tokens。

        参数：
            - **text** (TextInput) - 要编码的序列。
            - **pair** (str, 可选) - 与第一个序列一起编码的第二个序列。默认值： ``None`` 。
            - **add_special_tokens** (bool, 可选) - 是否添加与相应模型关联的特殊词元。默认值： ``False`` 。
            - **kwargs** (Any, 可选) - 这些参数将被传递给底层的具体模型编码方法。详见[`~PreTrainedTokenizerBase.__call__`]。

        返回：
            `tokenized_text`，类型为 `List[str]` 的词元列表。
