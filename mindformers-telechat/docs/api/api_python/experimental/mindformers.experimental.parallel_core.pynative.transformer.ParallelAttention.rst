mindformers.experimental.parallel_core.pynative.transformer.ParallelAttention
=============================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.transformer.ParallelAttention(config, layer_number, attention_type=1, attn_mask_type=1)

    该类表示并行注意力机制。它可以处理不同的注意力类型，并且可以使用各种参数进行配置。

    参数：
        - **config** (dict) - 一个配置字典，提供了并行注意力机制的各种设置。
        - **layer_number** (int) - 该transformer层在整个transformer块中的索引。
        - **attention_type** (int) - 注意力类型。支持1为self_attn，2为cross_attn，默认值： ``1`` 。
        - **attn_mask_type** (int) - 注意力mask类型。支持1为padding，2为causal，默认值： ``1`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - attention掩码矩阵，形状为 :math:`(B, N, S_q, S_k)`。
        - **encoder_output** (Tensor) - 用于交叉注意力的编码器输出张量。默认值： ``None``。
        - **inference_params** (Tensor) - 推理参数的张量，当前不支持该参数。默认值： ``None``。
        - **rotary_pos_emb** (Tensor) - 旋转位置嵌入张量。默认值： ``None``。

    输出：
        - **output** (Tensor) - 输出张量形状为 :math:`(B, S, H)`。
        - **bias** (Tensor) - 可训练的偏置参数。

    异常：
        - **ValueError** - 如果 `attention_type` 既不是 1 也不是 2。
        - **NotImplementedError** - 如果 `attention_type` 是 2 并且 `config` 中的 `group_query_attention` 是 true。
        - **ValueError** - 如果 `config` 中的 `hidden_size` 不等于 `config` 中的 `kv_hidden_size` 并且 `attention_type` 是 2。
        - **NotImplementedError** - 如果 `inference_params` 不是 None。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst