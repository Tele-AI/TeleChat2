mindformers.models.LlamaTokenizer
================================================

.. py:class:: mindformers.models.LlamaTokenizer(vocab_file, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>", sp_model_kwargs: Optional[Dict[str, Any]]=None, add_bos_token=True, add_eos_token=False, clean_up_tokenization_spaces=False, legacy=True, **kwargs)

    基于字节级字节对编码（Byte-Pair-Encoding）构建Llama分词器。
    默认未设置填充词元，因为原始模型中没有填充词元。

    参数：
        - **vocab_file** (str) - 词汇文件的路径。
        - **unk_token** (Union[str, AddedToken], 可选) - 未知词元。不在词汇表中的词元将被设置为此词元。默认值： ``"<unk>"`` 。
        - **bos_token** (Union[str, AddedToken], 可选) - 预训练时使用的序列开始词元。可以用作序列分类器词元。默认值： ``"<s>"`` 。
        - **eos_token** (Union[str, AddedToken], 可选) - 序列结束词元。默认值： ``"</s>"`` 。
        - **pad_token** (Union[str, AddedToken], 可选) - 用于批量处理时使词元数组大小相同的特殊词元。然后在注意力机制或损失计算中将被忽略。默认值： ``"<unk>"`` 。
        - **sp_model_kwargs** (Dict[str, Any], 可选) - 将传递给 `SentencePieceProcessor.__init__()` 方法。可以使用 `Python wrapper for SentencePiece <https://github.com/google/sentencepiece/tree/master/python>`_ 设置以下参数。默认值： ``None`` ，将传入一个空字典。
        - **add_bos_token** (bool, 可选) - 是否在序列开始处添加 `bos_token` 。默认值： ``True`` 。
        - **add_eos_token** (bool, 可选) - 是否在序列末尾添加 `eos_token` 。默认值： ``False`` 。
        - **clean_up_tokenization_spaces** (bool, 可选) - 解码后是否清理空格。清理包括去除如额外空格等潜在的符号。默认值： ``False`` 。
        - **use_default_system_prompt** (bool, 可选) - 是否使用Llama的默认系统提示。默认值： ``False`` 。
        - **spaces_between_special_tokens** (bool, 可选) - 是否在特殊词元之间添加空格。默认值： ``False`` 。
        - **legacy** (bool, 可选) - 是否使用分词器的 `legacy` 行为。默认值： ``True`` 。

    返回：
        `LlamaTokenizer` 实例。

    .. py:method:: build_inputs_with_special_tokens(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None)

        当前为将特殊词元插入到输入标识符中。

        参数：
            - **token_ids_0** (List[int]) - 第一组词元ID。
            - **token_ids_1** (List[int], 可选) - 第二组词元ID。默认值： ``None`` 。

        返回：
            插入特殊词元后的词元ID列表。

    .. py:method:: create_token_type_ids_from_sequences(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None)

        使用传递的两个序列创建一个掩码，用于序列对分类任务。一个ALBERT序列对掩码的格式如下：

        .. code-block::

            0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
            |       序列1       |       序列2       |

        如果 `token_ids_1` 为 None，则只返回掩码的第一部分（ ``0`` 序列）。

        参数：
            - **token_ids_0** (List[int]) - 第一个ID列表。
            - **token_ids_1** (List[int], 可选) - 序列对的可选第二个ID列表。默认值： ``None`` 。

        返回：
            一个根据给定序列由整数 ``0`` 和 ``1`` 组成的列表，其中 ``0`` 表示词元来自序列 `token_ids_0`， ``1`` 表示词元来自序列 `token_ids_1` 。

    .. py:method:: get_special_tokens_mask(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens:bool=False)

        从还未添加特殊词元的词元列表中检索序列ID。在使用分词器的 `prepare_for_model` 方法添加特殊词元时调用此方法。

        参数：
            - **token_ids_0** (List[int]) - ID列表。
            - **token_ids_1** (List[int], 可选) - 序列对的可选第二个ID列表。默认值： ``None`` 。
            - **already_has_special_tokens** (bool, 可选) - 词元列表是否已经按照模型格式添加了特殊词元。默认值： ``False`` 。

        返回：
            一个由整数 ``0`` 和 ``1`` 组成的列表，其中 ``1`` 表示特殊词元， ``0`` 表示序列词元。

