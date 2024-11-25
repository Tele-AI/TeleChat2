mindformers.core.PolynomialWithWarmUpLR
=======================================

.. py:class:: mindformers.core.PolynomialWithWarmUpLR(learning_rate: float, total_steps: int, warmup_steps: int = None, lr_end: float = 1e-7, power: float = 1.0, warmup_lr_init: float = 0., warmup_ratio: float = None, decay_steps: int = None, **kwargs)

    带有预热阶段的多项式衰减学习率。

    在训练的初期，学习率从一个较低的初始值 :math:`\eta_{\text{warmup}}` 逐步上升到初始学习率 :math:`\eta_{\text{start}}` 。预热阶段的学习率随步数 :math:`t` 变化的公式如下：

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{start}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    其中， :math:`\text{warmup\_steps}` 为预热阶段的总步数。

    在预热阶段结束后，学习率按照多项式函数逐渐衰减到设定的最终学习率 :math:`\eta_{\text{end}}` 。学习率在总步数 :math:`\text{total\_steps}` 中的变化可以通过以下公式表示：

    .. math::
        \eta_t = \eta_{\text{end}} + (\eta_{\text{start}} - \eta_{\text{end}}) \times \left(1 - \frac{t - \text{warmup_steps}}{\text{decay_steps}}\right)^{\text{power}}

    其中， :math:`\text{power}` 是多项式的幂次，用于控制衰减速度。

    该学习率适用于在训练初期需要稳定学习率并在训练后期逐渐降低学习率的场景。它通过在初期防止梯度爆炸并在后期降低学习率，帮助模型在收敛时获得更好的泛化性能。

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **total_steps** (int) - 总训练步数。
        - **warmup_steps** (int) - 预热阶段的步数。
        - **lr_end** (float) - 学习率的最终值。默认值： ``1e-7`` 。
        - **power** (float) - 多项式的幂次。默认值： ``1.0`` 。
        - **warmup_lr_init** (float) - 预热阶段的初始学习率。默认值： ``0.`` 。
        - **warmup_ratio** (float) - 预热阶段所占总训练步数的比例。默认值： ``None`` 。
        - **decay_steps** (int) - 衰减阶段的步数，必须小于 `total_steps - warmup_steps`。如果值为 ``None`` ，则衰减步数将为 `total_steps - warmup_steps`。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 当前训练步数。

    输出：
        学习率。
