mindformers.models.ChatGLM3Tokenizer
=========================================================================

.. py:class:: mindformers.models.ChatGLM3Tokenizer(vocab_file, bos_token='<sop>', eos_token='<eop>', end_token='</s>', mask_token='[MASK]', gmask_token='[gMASK]', pad_token='<pad>', unk_token='<unk>',*kwargs)

    构造一个基于Byte-Pair-Encoding的ChatGLM3模型分词器。

    参数：
        - **vocab_file** (str) - 对应词表的路径。
        - **bos_token** (Union[str, tokenizers.AddedToken], 可选) - 在预训练期间使用的序列开始的标记，可以用作序列分类标记。默认值： `"<sop>"` 。
        - **eos_token** (Union[str, tokenizers.AddedToken], 可选) - 序列结束token，标识输入输出序列的结束位置。默认值： `"</s>"` 。
        - **end_token** (Union[str, tokenizers.AddedToken], 可选) - 结束token，标识多个连续序列的最终结束位置。默认值： `"</s>"` 。
        - **mask_token** (Union[str, tokenizers.AddedToken], 可选) - 掩码token。默认值： `"[MASK]"` 。
        - **gmask_token** (Union[str, tokenizers.AddedToken], 可选) - 特殊的掩码token。默认值： `"[gMASK]"` 。
        - **pad_token** (Union[str, tokenizers.AddedToken], 可选) - 用于使tokens数组大小相同以便进行批处理的特殊标记，然后将被注意力机制或损失计算忽略。默认值： `"<pad>"` 。
        - **unk_token** (Union[str, tokenizers.AddedToken], 可选) - 不存在的token。默认值： `"<unk>"` 。
        - **kwargs** - 其它传递到Tokenizer基类的参数。

    返回：
        `ChatGLM3Tokenizer` 实例。
