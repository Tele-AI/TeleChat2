mindformers.core.EmF1Metric
===========================

.. py:class:: mindformers.core.EmF1Metric()

    EmF1Metric 用于计算每个示例的 Em 和 F1 分数，用于评估模型在预测任务中的性能。

    Em 分数: Em 分数表示预测值与标签值在忽略标点符号的情况下完全匹配的准确度。例如，问题是 "河南的省会是哪里？"，标签是 "郑州市"：

    当预测为 "郑州市" 时，Em 分数为 100；
    当预测为 "郑州市。" 时，Em 分数为 100；
    当预测为 "郑州" 时，Em 分数为 0。

    F1 分数: F1 分数是基于精度（precision）和召回率（recall）的调和平均数，计算公式如下：

    .. math::
        F1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}

    其中，精度和召回率的计算方式为：

    .. math::
        \text{precision} = \frac{\text{lcs_length}}{\text{len(prediction_segment)}}, \quad \text{recall} = \frac{\text{lcs_length}}{\text{len(label_segment)}}

    在上述公式中， :math:`\text{lcs_length}` 表示最长公共子序列的长度。

    计算过程:

    - 首先，计算预测文本与标签文本的最长公共子序列（LCS），用来衡量预测和标签之间的匹配程度。

    - 然后，通过精度和召回率的公式计算对应的值。

    - 最后，使用 F1 分数公式计算最终的 F1 值。

    该评估指标能够全面衡量模型的准确性和完整性，为模型的优化和调试提供数据支持。
