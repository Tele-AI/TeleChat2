mindformers.core.LearningRateWiseLayer
======================================

.. py:class:: mindformers.core.LearningRateWiseLayer(base_lr, lr_scale)

    学习率分层模块。

    这种方法允许每层根据其特定需求调整学习率，从而实现更高效和有效的训练。每层的学习率由基准学习率与该层特定的缩放因子调节决定。

    最初，每层的学习率基于线性缩放策略设定：

    .. math::
        \eta_{t,l} = \eta_{\text{base}} \times \alpha_l

    其中 :math:`\eta_{t,l}` 是时间 :math:`t` 时层 :math:`l` 的学习率， :math:`\eta_{\text{base}}` 是基准学习率， :math:`\alpha_l` 是层 :math:`l` 的缩放因子。

    随着训练的进行，每层的学习率按照以下余弦退火调度进行调整：

    .. math::
        \eta_{t,l} = \eta_{\text{end}} + \frac{1}{2}(\eta_{\text{base}} \times \alpha_l - \eta_{\text{end}})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    其中 :math:`T_{cur}` 是自学习率上次重置以来完成的epoch数量， :math:`T_{max}` 是下一次重置前的总epoch数。 :math:`\eta_{\text{end}}` 代表训练结束时的最小学习率。

    参数：
        - **base_lr** (:class:`mindspore.nn.learning_rate_schedule.LearningRateSchedule`) - 基准学习率调度器。
        - **lr_scale** (float) - 学习率缩放值。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
