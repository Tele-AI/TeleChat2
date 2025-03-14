mindformers.core.EntityScore
============================

.. py:class:: mindformers.core.EntityScore()

    评估预测实体相对于真实标签的精确率、召回率和 F1 分数。

    数学上，这些指标定义如下：

    精确率 (Precision)：衡量预测的实体中正确预测的比例。

    .. math::
        \text{Precision} = \frac{\text{正确预测的实体数量}}{\text{预测的实体总数量}}

    召回率 (Recall)：衡量实际存在的实体中被正确预测的比例。

    .. math::
        \text{Recall} = \frac{\text{正确预测的实体数量}}{\text{实际实体总数量}}

    F1 分数 (F1 Score)：精确率和召回率的调和平均值，提供两者之间的平衡。

    .. math::
        \text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

    .. py:method:: clear()

        清除局部评估结果。

    .. py:method:: eval()

        计算评估结果。

        返回：
            评估结果字典，包含实体相对于真实标签的精确率、召回率和 F1 分数。

    .. py:method:: update(*inputs)

        更新局部评估结果。

        参数：
            - **\*inputs** (List) - 逻辑值和标签。其中逻辑值是形状为 :math:`[N,C]` 的张量，数据类型为Float16或Float32；标签是形状为 :math:`[N,]` 的张量，数据类型为Int32或Int64。其中 :math:`N` 为批次大小， :math:`C` 为实体类型总数。