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
