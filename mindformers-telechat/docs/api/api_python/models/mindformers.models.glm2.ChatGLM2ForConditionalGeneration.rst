mindformers.models.glm2.ChatGLM2ForConditionalGeneration
=========================================================================

.. py:class:: mindformers.models.glm2.ChatGLM2ForConditionalGeneration(config: ChatGLM2Config, **kwargs)

    在线计算并提供执行ChatGLM2训练时的损失值和逻辑值。

    参数：
        - **config** (ChatGLM2Config) - ChatGLM2模型的配置。
        - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

    输入：
        - **input_ids** (Tensor) - 一个分词后的输入数据张量，它是32位整数类型，shape为： `(batch, seq_length)` 。默认值： ``None`` 。
        - **labels** (Tensor) - 一个分词后的标签数据张量，它是32位整数类型，shape为： `(batch, seq_length)` 。默认值： ``None`` 。
        - **input_position** (Tensor) - 当前位置，在推理使使用。默认值： ``None`` 。
        - **position_ids** (Tensor) - 保留参数，不使用。默认值： ``None`` 。
        - **attention_mask** (Tensor) - 保留参数，不使用。默认值： ``None`` 。
        - **input_embeds** (Tensor) - 保留参数，不使用。默认值： ``None`` 。
        - **init_reset** (bool) - shape为[1]的bool张量，用于清除增量推理中之前的键参数和值参数。默认值： ``None`` 。
        - **batch_valid_length** (Tensor) - 在增量推理中，用于上一步计算索引的张量。它是32位整数类型，shape为 `[batch_size]` 。默认值： ``None`` 。
        - **prefix_key_values** (Tensor) - 在正常的键值对之前添加的一组额外的键值对。这些前缀键值对可以用来捕获长期依赖关系或提供先验知识，从而帮助模型更好地理解和生成序列。默认值： ``None`` 。
        - **block_tables** (Tensor[int64]) - 存储每个序列的映射表。默认值： ``None`` 。
        - **slot_mapping** (Tensor[int32]) - 存储序列缓存的物理槽索引。默认值： ``None`` 。
        - **batch_index**  (Tensor) - 保留参数，不使用。默认值： ``None`` 。
        - **zactivate_len** (Tensor) - 保留参数，不使用。默认值： ``None`` 。

    输出：
        outputs(Tensor)，包括在线损失值或者逻辑值、预测文本序列、输入掩码。
