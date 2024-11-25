mindformers.models.PreTrainedTokenizerFast
==========================================

.. py:class:: mindformers.models.PreTrainedTokenizerFast(*args, **kwargs)

    快速分词器的基类，封装了 HuggingFace 分词器库。

    处理所有分词和特殊词元的共享方法，以及下载/缓存/加载预训练分词器的方法，还包括向词汇表中添加词元的方法。

    此类还在所有分词器之上统一处理了添加的词元，因此我们无需处理各种底层字典结构（BPE、sentencepiece等）的特定词汇增强方法。

    参数：
        - **model_max_length** (int, 可选) - transformer模型输入的最大长度（以词元数量计）。当通过 `from_pretrained()` 加载分词器时，此值将设置为 `max_model_input_sizes` 中存储的关联模型的值。默认值： ``1e-30`` 。
        - **padding_side** (str, 可选) - 模型应该在哪一侧应用填充。应从 ['right', 'left'] 中选择。默认值从同名类属性中选择。
        - **truncation_side** (str, 可选) - 模型应该在哪一侧应用截断。应从 ['right', 'left'] 中选择。默认值从同名类属性中选择。
        - **chat_template** (str, 可选) - 将用于格式化聊天消息列表的Jinja模板字符串。默认值： ``"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"`` 。
        - **model_input_names** (List[str], 可选) - 模型前向传递接受的输入列表（如 "token_type_ids" 或 "attention_mask" ）。默认值从同名类属性中选择。默认值： ``None`` 。
        - **bos_token** (Union[str, tokenizers.AddedToken], 可选) - 表示句子开始的特殊词元。将关联到 ``self.bos_token`` 和 ``self.bos_token_id`` 。默认值： ``None`` 。
        - **eos_token** (Union[str, tokenizers.AddedToken], 可选) - 表示句子结束的特殊词元。将关联到 ``self.eos_token`` 和 ``self.eos_token_id`` 。默认值： ``None`` 。
        - **unk_token** (Union[str, tokenizers.AddedToken], 可选) - 表示词汇表外词元的特殊词元。将关联到 ``self.unk_token`` 和 ``self.unk_token_id`` 。默认值： ``None`` 。
        - **sep_token** (Union[str, tokenizers.AddedToken], 可选) - 在同一输入中分隔两个不同句子的特殊词元（例如BERT使用）。将关联到 ``elf.sep_token`` 和 ``self.sep_token_id`` 。默认值： ``None`` 。
        - **pad_token** (Union[str, tokenizers.AddedToken], 可选) - 用于使词元数组大小相同，以便批处理的特殊词元。注意机制或损失计算将忽略它。将关联到 ``self.pad_token`` 和 ``self.pad_token_id`` 。默认值： ``None`` 。
        - **cls_token** (Union[str, tokenizers.AddedToken], 可选) - 表示输入类的特殊词元（例如BERT使用）。将关联到 ``self.cls_token`` 和 ``self.cls_token_id`` 。默认值： ``None`` 。
        - **mask_token** (Union[str, tokenizers.AddedToken], 可选) - 表示掩码词元的特殊词元（用于掩码语言建模预训练目标，如BERT）。将关联到 ``self.mask_token`` 和 ``self.mask_token_id`` 。默认值： ``None`` 。
        - **additional_special_tokens** (Union[tuple, list, tokenizers.AddedToken], 可选) - 一组额外的特殊词元。在这里添加它们以确保在设置 ``skip_special_tokens`` 为 ``True`` 时跳过它们。如果它们不是词汇表的一部分，将在词汇表的末尾添加。默认值： ``None`` 。
        - **clean_up_tokenization_spaces** (bool, 可选) - 是否清理在分词过程中添加的空格。默认值： ``True`` 。
        - **split_special_tokens** (bool, 可选) - 是否在分词过程中拆分特殊词元。传递将影响分词器的内部状态。默认行为是不拆分特殊词元。这意味着如果 ``<s>`` 是 ``bos_token`` ，则 ``tokenizer.tokenize("<s>") = ['<s>']`` 。否则，如果 ``split_special_tokens = True`` ，则 ``tokenizer.tokenize("<s>")`` 会得到 ``['<','s', '>']`` 。默认值： ``False`` 。
        - **tokenizer_object** (tokenizers.Tokenizer) - 一个 ``tokenizer.Tokenizer`` 对象，用于实例化。
        - **tokenizer_file** (str) - 一个本地JSON文件的路径，该文件代表一个之前序列化的 ``tokenizer.Tokenizer`` 对象。

    返回：
        PreTrainedTokenizerFast类实例。

    .. py:method:: added_tokens_decoder()
        :classmethod:

        返回词汇表中作为索引到AddedToken的字典形式的添加的词元。

        返回：
            dict，添加的词元。

    .. py:method:: added_tokens_encoder()
        :classmethod:

        返回从字符串到索引的排序映射。为了性能优化，添加的词元编码器在慢速分词器的 `self._added_tokens_encoder` 中被缓存。

        返回：
            dict，添加的词元。

    .. py:method:: convert_ids_to_tokens(ids: Union[int, List[int]], skip_special_tokens: bool = False)

        使用词汇表和添加的词元将单个索引或索引序列转换为词元或词元序列。

        参数：
            - **ids** (Union[int, List[int]]) - 要转换的词元id或词元id序列。
            - **skip_special_tokens** (bool, 可选) - 是否在解码时去除特殊词元。默认值： ``False`` 。

        返回：
            `str` 或 `List[str]`，解码后的词元或词元序列。

    .. py:method:: convert_tokens_to_ids(tokens: Union[str, List[str]])

        使用词汇表将词元字符串（或词元序列）转换为单个整数id（或id序列）。

        参数：
            - **tokens** (Union[str, List[str]]) - 要转换为词元id的一个或多个词元。

        返回：
            `int` 或 `List[int]`，词元id或词元id列表。

    .. py:method:: get_added_vocab()

        返回词汇表中作为词元到索引的字典形式的添加的词元。

        返回：
            dict，添加的词元。

    .. py:method:: num_special_tokens_to_add(pair: bool = False)

        返回在编码带有特殊词元的序列时添加的词元数量。

        .. note::
            此操作通过编码虚拟输入来检查添加的词元数量，效率不高。不建议将此操作放在训练循环中。

        参数：
            - **pair** (bool, 可选) - 是否针对序列对计算添加的词元数量。默认值： ``False`` 。

        返回：
            `int`，添加到序列中的特殊词元数量。

    .. py:method:: set_truncation_and_padding(padding_strategy: PaddingStrategy, truncation_strategy: TruncationStrategy, max_length: int, stride: int, pad_to_multiple_of: Optional[int])

        定义快速分词器的截断和填充策略，并在之后恢复分词器设置。

        参数：
            - **padding_strategy** (PaddingStrategy) - 将应用于输入的填充类型。
            - **truncation_strategy** (TruncationStrategy) - 将应用于输入的截断类型。
            - **max_length** (int) - 序列的最大大小。
            - **stride** (int) - 处理溢出时使用的步幅。
            - **pad_to_multiple_of** (int, 可选) - 如果设置，将序列填充到提供值的倍数。默认值： ``None`` 。

    .. py:method:: train_new_from_iterator(text_iterator, vocab_size, length=None, new_special_tokens=None, special_tokens_map=None, **kwargs)

        使用与当前分词器相同的特殊词元或分词流程的默认设置，在新语料库上训练分词器。

        参数：
            - **text_iterator** (list) - 训练语料库。应该是文本批次的生成器，例如如果您的所有数据都在内存中，可以是文本列表的列表。
            - **vocab_size** (int) - 您想要的分词器的词汇表大小。
            - **length** (int, 可选) - 迭代器中的序列总数，用于提供有意义的进度跟踪。默认值： ``None`` 。
            - **new_special_tokens** (Union[list, AddedToken], 可选) - 要添加到您正在训练的分词器的新特殊词元列表。默认值： ``None`` 。
            - **special_tokens_map** (dict, 可选) - 如果您想重命名此分词器使用的某些特殊词元，请在此参数中传递旧特殊词元名称到新特殊词元名称的映射。默认值： ``None`` 。
            - **kwargs** (Any, 可选) - 用于标记化的关键字参数。

        返回：
            [`PreTrainedTokenizerFast`]，与原始分词器类型相同、在 `text_iterator` 上训练的新分词器。

