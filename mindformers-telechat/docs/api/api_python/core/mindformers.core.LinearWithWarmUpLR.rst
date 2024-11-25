mindformers.core.LinearWithWarmUpLR
===================================

.. py:class:: mindformers.core.LinearWithWarmUpLR(learning_rate: float, total_steps: int, warmup_steps: int = None, warmup_lr_init: float = 0., warmup_ratio: float = None, **kwargs)

    线性预热学习率。

    该学习率使用线性预热策略来逐步增加每个参数组的学习率，并在预热阶段结束后按线性方式调整学习率。

    在预热阶段，学习率从一个较小的初始值线性增加到基准学习率，公式如下：

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    其中， :math:`\eta_{\text{warmup}}` 是预热阶段的初始学习率， :math:`\eta_{\text{base}}` 是预热阶段结束后的基准学习率。

    预热阶段结束后，学习率将按照以下线性调度公式进行调整：

    .. math::
        \eta_t = \eta_{\text{base}} - t \times \frac{\eta_{\text{base}} - \eta_{\text{end}}}{\text{total_steps} - \text{warmup_steps}}

    其中， :math:`\eta_{\text{end}}` 是训练结束时的最小学习率， :math:`\text{total_steps}` 是总的训练步数， :math:`\text{warmup_steps}` 是预热阶段的步数。

    这种方法允许通过线性预热来平滑地增加学习率，然后在剩余的训练过程中逐步降低学习率，以提高训练的稳定性和效果。

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **total_steps** (int) - 总步数。
        - **warmup_steps** (int) - 预热步骤数。默认值： ``None`` 。
        - **warmup_lr_init** (float) - 预热阶段的初始学习率。默认值： ``0.`` 。
        - **warmup_ratio** (float) - 用于预热的总训练步数比例。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
