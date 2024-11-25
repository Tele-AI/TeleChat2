mindformers.core.PerplexityMetric
=================================

.. py:class:: mindformers.core.PerplexityMetric()

    困惑度定义为模型对测试集中每个词分配的负对数概率的指数平均值。对于一个词序列 :math:`W = (w_1, w_2, \ldots, w_N)` ，困惑度 (PP) 可以表示为：

    .. math::
        PP(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, \ldots, w_N)}}

    其中， :math:`P(w_1, w_2, \ldots, w_N)` 表示该序列在模型下的概率。

    在实际应用中，困惑度可以重写为：

    .. math::
        PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, w_2, \ldots, w_{i-1})\right)

    该公式表明，较低的困惑度意味着语言模型性能更好，因为这表示模型对实际的词序赋予了更高的概率。
