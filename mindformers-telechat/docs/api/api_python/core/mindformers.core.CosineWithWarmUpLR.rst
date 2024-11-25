mindformers.core.CosineWithWarmUpLR
===================================

.. py:class:: mindformers.core.CosineWithWarmUpLR(learning_rate: float, warmup_steps: int = 0, total_steps: int = None, num_cycles: float = 0.5, lr_end: float = 0., warmup_lr_init: float = 0., warmup_ratio: float = None, decay_steps: int = None, **kwargs)

    余弦预热学习率。

    该学习率使用余弦退火调度结合预热步骤来设置每个参数组的学习率。最初，学习率在预热阶段线性增加，然后遵循余弦函数逐渐衰减。

    在预热阶段，学习率从一个较小的初始值增加到基准学习率，公式如下：

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    其中， :math:`\eta_{\text{warmup}}` 是初始学习率， :math:`\eta_{\text{base}}` 是预热阶段结束后的学习率。

    预热阶段结束后，学习率将按照余弦衰减公式进行衰减：

    .. math::
        \eta_t = \eta_{\text{end}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{end}})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    其中， :math:`T_{cur}` 是自预热阶段结束以来的epoch数量， :math:`T_{max}` 是下次重启前的总epoch数。

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **warmup_steps** (int) - 预热步骤数。默认值： ``None`` 。
        - **total_steps** (int) - 总的训练步骤数。默认值： ``None`` 。
        - **num_cycles** (float) - 余弦调度中的周期数量（默认情况下为半个周期，从最大值递减至 0）。默认值： ``0.5`` 。
        - **lr_end** (float) - 学习率的最终值。默认值： ``0.0`` 。
        - **warmup_lr_init** (float) - 预热阶段的初始学习率。默认值： ``0.0`` 。
        - **warmup_ratio** (float) - 预热阶段占总训练步骤的比例。默认值： ``None`` 。
        - **decay_steps** (int) - 衰减步骤的数量。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
