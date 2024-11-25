mindformers.generation.GenerationConfig
=======================================

.. py:class:: mindformers.generation.GenerationConfig(**kwargs)

    保存生成任务配置的类。

    某些参数有特定的功能，有关详细信息，请参见下表：

    +---------------------------+------------------------------+
    | 功能分类                  |  配置参数                    |
    +===========================+==============================+
    | 控制输出长度的参数        |  max_length                  |
    |                           +------------------------------+
    |                           |  max_new_tokens              |
    |                           +------------------------------+
    |                           |  min_length                  |
    |                           +------------------------------+
    |                           |  min_new_tokens              |
    +---------------------------+------------------------------+
    | 控制所用生成策略的参数    |  do_sample                   |
    |                           +------------------------------+
    |                           |  use_past                    |
    +---------------------------+------------------------------+
    | 模型输出逻辑操作参数      |  temperature                 |
    |                           +------------------------------+
    |                           |  top_k                       |
    |                           +------------------------------+
    |                           |  top_p                       |
    |                           +------------------------------+
    |                           |  repetition_penalty          |
    |                           +------------------------------+
    |                           |  encoder_repetition_penalty  |
    |                           +------------------------------+
    |                           |  renormalize_logits          |
    +---------------------------+------------------------------+
    | 定义generate输出变量的参数|  output_scores               |
    |                           +------------------------------+
    |                           |  output_logits               |
    |                           +------------------------------+
    |                           |  return_dict_in_generate     |
    +---------------------------+------------------------------+
    | 可在生成时使用的特殊词元  |  pad_token_id                |
    |                           +------------------------------+
    |                           |  bos_token_id                |
    |                           +------------------------------+
    |                           |  eos_token_id                |
    +---------------------------+------------------------------+

    参数：
        - **max_length** (int, 可选) - 生成的词元可以具有的最大长度。对应于输入提示符的长度 + `max_new_tokens` 。如果也设置了 `max_new_tokens` ，则其效果将被 `max_new_tokens` 覆盖。默认值： ``20`` 。
        - **max_new_tokens** (int, 可选) - 要生成的词元的最大数目，忽略提示符中的词元数目。默认值： ``None`` 。
        - **min_length** (int, 可选) - 要生成的序列的最小长度。对应输入提示符的长度 + `min_new_tokens` 。如果也设置了 `min_new_tokens` ，则其效果将被 `min_new_tokens` 覆盖。默认值： ``0`` 。
        - **min_new_tokens** (int, 可选) - 要生成的最小词元数，忽略提示符中的词元数。默认值： ``None`` 。
        - **do_sample** (bool, 可选) - 是否使用采样编码。 ``True`` 表示使用采样编码， ``False`` 代表使用贪婪解码。默认值： ``False`` 。
        - **use_past** (bool, 可选) - 模型是否应使用过去最后一个键/值注意（如果适用于模型）来加快解码速度。默认值： ``False`` 。
        - **temperature** (float, 可选) - 用于调节下一个词元概率的值。默认值： ``1.0`` 。
        - **top_k** (int, 可选) - 为top-k筛选保留的最高概率词汇词元的数量。默认值： ``50`` 。
        - **top_p** (float, 可选) - 如果设置为 ``float < 1`` ，则仅保留概率加起来等于或更高的最小最可能标记集以供生成。默认值： ``1.0`` 。
        - **repetition_penalty** (float, 可选) - 重复惩罚的参数。1.0表示没有处罚。参考 `此内容 <https://arxiv.org/pdf/1909.05858.pdf>`_ 。默认值： ``1.0`` 。
        - **encoder_repetition_penalty** (float, 可选) - encoder_repeation_ppenalty的参数。对不在原始输入中的序列的指数惩罚。1.0表示没有惩罚。默认值： ``1.0`` 。
        - **renormalize_logits** (bool, 可选) - 是否在应用所有logits处理器或扭曲器（包括自定义处理器）后重新规范化logits。强烈建议将此标志设置为 ``True`` ，因为搜索算法假设分数logit是标准化的，但一些logit处理器或扭曲器会破坏标准化。默认值： ``False`` 。
        - **output_scores** (bool, 可选) - 是否在softmax之前返回预测分数。默认值： ``False`` 。
        - **output_logits** (bool, 可选) - 是否返回未处理的预测logit分数。默认值： ``False`` 。
        - **return_dict_in_generate** (bool, 可选) - 是否返回字典输出而不是具有output_ids的元组。默认值： ``False`` 。
        - **pad_token_id** (int, 可选) - padding词元的id。
        - **bos_token_id** (int, 可选) - beginning-of-sequence词元的id。
        - **eos_token_id** (Union[int, List[int]], 可选) - end-of-sequence词元的id。设置多个end-of-sequence词元。

    返回：
        GenerationConfig实例。