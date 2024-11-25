mindformers.experimental.parallel_core.pynative.VocabParallelCrossEntropy
=========================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.VocabParallelCrossEntropy()

    计算并行交叉熵损失。

    输入：
        - **vocab_parallel_logits** (Tensor) - 主干网络的输出。形状为 ``(N, C)`` 的张量。数据类型必须为float16或float32。
        - **target** (Tensor) - 样本的真值。形状为 ``(N,)`` 。
        - **label_smoothing** (Float) - 平滑因子，必须在范围 ``[0.0, 1.0)`` 内。

    输出：
        - **loss** (Tensor) - 对应的交叉熵损失。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst
