mindformers.models.LlamaTokenizerFast
========================================================

.. py:class:: mindformers.models.LlamaTokenizerFast(vocab_file=None, tokenizer_file=None, clean_up_tokenization_spaces=False, unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=True, add_eos_token=False, use_default_system_prompt=False, **kwargs)

    构建基于字节级字节对编码（Byte-Pair-Encoding）的 Llama 快速分词器。此分词器使用 ByteFallback 且没有标准化处理。

    参数：
        - **vocab_file** (str, 可选) - 包含必要词汇表的 `SentencePiece <https://github.com/google/sentencepiece>` 文件（通常具有 .model 扩展名），用于实例化分词器。默认值： ``None`` 。
        - **tokenizer_file** (str, 可选) - 包含加载分词器所需所有内容的 tokenizers 文件（通常具有 .json 扩展名）。默认值： ``None`` 。
        - **clean_up_tokenization_spaces** (bool, 可选) - 解码后是否清理空格，清理包括去除可能的如额外空格等人工制品。默认值： ``False`` 。
        - **unk_token** (Union[str, tokenizers.AddedToken], 可选) - 未知词元。不在词汇表中的词元将被设置为此词元。默认值： ``"<unk>"`` 。
        - **bos_token** (Union[str, tokenizers.AddedToken], 可选) - 预训练时使用的序列开始词元。可以用作序列分类器词元。默认值： ``"<s>"`` 。
        - **eos_token** (Union[str, tokenizers.AddedToken], 可选) - 序列结束词元。默认值： ``"</s>"`` 。
        - **add_bos_token** (bool, 可选) - 是否在序列开始处添加 `bos_token`。默认值： ``True`` 。
        - **add_eos_token** (bool, 可选) - 是否在序列末尾添加 `eos_token`。默认值： ``False`` 。
        - **use_default_system_prompt** (bool, 可选) - 是否使用 Llama 的默认系统提示。默认值： ``False`` 。

    .. note::
        如果您想更改 `bos_token` 或 `eos_token`，请确保在初始化模型时指定它们，或者调用 `tokenizer.update_post_processor()` 确保正确地完成后处理（否则编码序列的第一个和最后一个词元的值将不正确）。

    返回：
        LlamaTokenizer 类实例。

    .. py:method:: build_inputs_with_special_tokens(token_ids_0, token_ids_1=None)

        将特殊词元插入到输入标识符中。此方法用于在序列的开头和结尾添加开始（BOS）和结束（EOS）词元。

        参数：
            - **token_ids_0** (List[int]) - 第一组词元ID。
            - **token_ids_1** (List[int], 可选) - 第二组词元ID。默认值： ``None`` 。

        返回：
            - 返回一个列表，其中包含在序列两端插入特殊词元后的词元ID。

    .. py:method:: save_vocabulary(save_directory: str, filename_prefix: Optional[str] = None)

        将词汇表保存到指定目录。此方法用于从慢速分词器中导出词汇表文件。

        参数：
            - **save_directory** (str) - 词汇表的保存目录。
            - **filename_prefix** (str, 可选) - 保存文件的前缀。默认值： ``None`` 。

        返回：
            - 返回一个元组，包含保存的词汇表文件的路径。

        异常：
            - **ValueError** - 如果无法从快速分词器保存词汇表，或者提供的保存目录不存在，将抛出此异常。

    .. py:method:: update_post_processor()

        更新底层的后处理函数，以使用当前的 `bos_token` 和 `eos_token`。

        异常：
            - **ValueError** - 如果设置了 `add_bos_token` 或 `add_eos_token` 但相应的 token 为 `None`，则抛出此异常。
