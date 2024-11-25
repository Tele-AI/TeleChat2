mindformers.experimental.parallel_core.pynative.RotaryEmbedding
===============================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.RotaryEmbedding(kv_channels, rotary_percent=1.0, rotary_interleaved=False, seq_len_interpolation_factor=None, rotary_base=10000)

    用于语言模型的旋转位置嵌入。

    参数：
        - **kv_channels** (int) - 多头注意力中的投影权重维度。从transformer的配置中获取。
        - **rotary_percent** (float, 可选) - 旋转位置编码中旋转维度的使用比例。默认值： ``1.0`` 。
        - **rotary_interleaved** (bool, 可选) - 是否以交错方式将旋转位置编码应用于输入维度。默认值： ``False`` 。目前暂不支持设置为 ``True`` 。
        - **seq_len_interpolation_factor** (float, 可选) - 对更长序列进行线性插值的比例。如果设置非None，则该值必须是大于1.0的浮点数。默认值： ``None`` 。
        - **rotary_base** (int, 可选)：旋转位置嵌入编码的基期。默认值： ``10000`` 。

    输入：
        - **max_seq_len** (int) - 输入的最大序列长度。
        - **offset** (int) - 位置编码偏移量。

    输出：
        - **emb** (Tensor) - 应用旋转位置编码后的嵌入向量。

    异常：
        - **NotImplementedError** – 当rotary_interleaved为True。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst
