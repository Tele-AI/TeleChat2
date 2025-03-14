mindformers.models.LlamaForCausalLM
=========================================================================

.. py:class:: mindformers.models.LlamaForCausalLM(config: LlamaConfig = None)

    在线计算并提供执行LLama训练时的损失值和逻辑值。

    参数：
        - **config** (LlamaConfig, 可选) - LLama模型的配置。默认值： ``None`` 。

    输入：
        - **input_ids** (Tensor) - 数据类型为Int64/Int32的词汇表中输入序列标记的索引，张量的形状为：:math:`(batch, seq\_length)`。
        - **labels** (Tensor, 可选) - 数据类型为Int64/Int32的输入标签，张量的形状为：:math:`(batch, seq\_length)` 。默认值： ``None`` 。
        - **input_position** (Tensor, 可选) - 输入的位置索引。在增量推理模式下，位置索引是数据类型为Int64/Int32的递增序列。张量的形状为：:math:`(batch, seq\_length)` 。默认值： ``None`` 。
        - **position_ids** (Tensor, 可选) - 输入的位置id，数据类型是Int64/Int32的递增序列。张量的形状为：:math:`(batch, seq\_length)` 。默认值： ``None`` 。
        - **attention_mask** (Tensor, 可选) - 输入句子填充掩码，其中0表示填充位置。数据类型Int64/Int32，张量的形状为：:math:`(batch, seq\_length)` 。默认值： ``None`` 。
        - **input_embeds** (Tensor, 可选) - 数据类型Float32/Float16的输入嵌入。张量的形状为：:math:`(batch, seq\_length, hidden_size)` 。默认值： ``None`` 。
        - **init_reset** (Tensor, 可选) - 数据类型为Bool，表示是否清除增量推理中之前的键参数和值参数。仅当use_past为True时有效。关于use_past的定义可以参考[GenerationConfig]()。张量的形状为：:math:`(1)`。默认值： ``Tensor([True])`` 。
        - **batch_valid_length** (Tensor, 可选) - 数据类型为Int32，表示批次中每个序列已经计算过的长度。张量的形状为：:math:`(batch_size)` 。默认值： ``None`` 。
        - **batch_index** (Tensor, 可选) - 弃用参数。将会在未来删除。默认值： ``None`` 。
        - **zactivate_len** (Tensor, 可选) - 弃用参数。将会在未来删除。默认值： ``None`` 。
        - **block_tables** (Tensor, 可选) - 数据类型为Int64，存储每个序列的映射表。默认值： ``None`` 。
        - **slot_mapping** (Tensor, 可选) - 数据类型为Int32，存储词元缓存的物理槽索引。默认值： ``None`` 。
        - **prefix_keys_values** (Tensor, 可选) - 弃用参数。将会在未来删除。默认值： ``None`` 。
        - **llm_boost_inputs** (Tensor, 可选) - 弃用参数。将会在未来删除。默认值： ``None`` 。
        - **q_seq_lens** (Tensor, 可选) - 在并行解码中，query可能会被扁平化。分页注意力算子需要使用 `q_seq_lens` 获取长度信息。默认值： ``None`` 。
        - **loss_mask** (Tensor, 可选) - 数据类型为Float32/Int32，用于确定对应的token位置是否参与loss计算，如果值为 :math:`(1)` 则计算该位置的loss，:math:`(0)` 则不计算。默认值：``None`` 。
        - **gather_index** (Tensor, 可选) - 数据类型为Int32，用于获取每个sequence的最后一个token对应位置的隐向量计算logits。张量的形状为：:math:`(batch_size)` 。默认值： ``None`` 。
        - **seq_range** (Tensor, 可选) - 数据类型为Int32，用于指定每个sequence的实际长度。张量的形状为：:math:`(batch_size)` 。默认值： ``None`` 。

    输出：
        Tensor类型。如果是训练模式，输出的Tensor包含在线损失值；如果是推理模式，输出的Tensor包含逻辑值；如果是评测模式，输出的Tensor包含逻辑值、词元、输入掩码。