mindformers.models.glm2.ChatGLM4Tokenizer
=========================================================================

.. py:class:: mindformers.models.glm2.ChatGLM4Tokenizer(vocab_file, clean_up_tokenization_spaces=False, encode_special_tokens=False, eos_token='<|endoftext|>', pad_token='<|endoftext|>', **kwargs)

    构造一个基于Byte-Pair-Encoding的ChatGLM4模型分词器。

    参数：
        - **vocab_file** (str) - 对应词表的路径。
        - **clean_up_tokenization_spaces** (bool) - 是否清理掉多余的空格。默认值： ``False`` 。
        - **encode_special_tokens** (bool) - 是否清理特殊token。默认值： ``False`` 。
        - **eos_token** (str, tokenizers.AddedToken) - 序列结束标记。默认值： `"<|endoftext|>"` 。
        - **pad_token** (str, tokenizers.AddedToken) - 用于使tokens数组大小相同以便进行批处理的特殊标记，然后将被注意力机制或损失计算忽略。默认值： `"<|endoftext|>"` 。
        - **kwargs** - 其它传递到Tokenizer基类的参数。

    返回：
        `ChatGLM4Tokenizer` 实例。
