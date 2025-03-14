mindformers.core.PromptAccMetric
================================

.. py:class:: mindformers.core.PromptAccMetric()

    计算每个实体的提示准确率（prompt acc）。提示准确率是基于构建提示的文本分类准确度。准确的索引是具有最小困惑度的提示索引。

    1. 为该评估指标构建提示的方式如下：

       .. code-block::

           这是关于**体育**的文章：$passage
           这是关于**文化**的文章：$passage

    2. 计算基于提示生成的每个上下文的困惑度。困惑度是衡量概率分布或模型预测样本能力的指标。较低的困惑度表示模型能够很好地预测样本。公式如下：

       .. math::
          PP(W) = P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

       其中， :math:`w` 代表语料库中的词。

    3. 通过选择困惑度最小的提示索引来计算分类结果。

    4. 计算正确分类的数量和样本总数，并计算准确率，公式如下：

       .. math::
          \text{accuracy} = \frac{\text{correct_sample_nums}}{\text{total_sample_nums}}

    .. py:method:: clear()

        清除局部评估结果。

    .. py:method:: eval()

        计算评估结果。

        返回：
            评估结果字典，包含 Acc 分数。

    .. py:method:: update(*inputs)

        更新局部评估结果。

        参数：
            - **\*inputs** (List) - 逻辑值、输入索引、输入掩码和标签。其中逻辑值是形状为 :math:`[N,C,S,W]` 的张量，数据类型为Float16或Float32；输入索引、输入掩码和标签是形状为 :math:`[N*C,S]` 的张量，数据类型为Int32或Int64。其中 :math:`N` 为批次大小， :math:`C` 为实体类型总数， :math:`S` 为序列长度， :math:`W` 为词表大小。