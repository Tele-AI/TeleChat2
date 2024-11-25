mindformers.core.ConstantWarmUpLR
=================================

.. py:class:: mindformers.core.ConstantWarmUpLR(learning_rate: float, warmup_steps: int = None, warmup_lr_init: float = 0., warmup_ratio: float = None, total_steps: int = None, **kwargs)

    恒定预热学习率。

    该学习率在预热阶段保持恒定的学习率。此策略特别适用于需要在训练初期使用稳定、较低的学习率，以避免如梯度爆炸等问题，然后再过渡到主要学习率调度的场景。

    在预热阶段，学习率保持在一个固定的值，记为 :math:`\eta_{\text{warmup}}` 。预热阶段的学习率公式为：

    .. math::
        \eta_t = \eta_{\text{warmup}}

    其中， :math:`\eta_{\text{warmup}}` 是在预热步骤中应用的固定学习率， :math:`t` 代表当前步骤。

    在预热阶段结束后，学习率过渡到主要学习率，记为 :math:`\eta_{\text{main}}` 。过渡后的学习率公式为：

    .. math::
        \eta_t = \eta_{\text{main}}

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **warmup_steps** (int) - 预热阶段的步数。默认值： ``None`` 。
        - **warmup_lr_init** (float) - 预热阶段的初始学习率。默认值： ``0.`` 。
        - **warmup_ratio** (float) - 预热阶段占总训练步数的比例。默认值： ``None`` 。
        - **total_steps** (int) - 总的预热步数。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
