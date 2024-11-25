mindformers.core.CosineAnnealingWarmRestarts
============================================

.. py:class:: mindformers.core.CosineAnnealingWarmRestarts(base_lr: float, t_0: int, t_mult: int = 1, eta_min: float = 0., **kwargs)

    使用余弦退火调度设置每个参数组的学习率。其中 :math:`\eta_{max}` 被设为初始学习率， :math:`T_{cur}` 表示自上次重启以来的epoch数量， :math:`T_{i}` 表示两次热重启之间的epoch数量，在SGDR中计算学习率：

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    当 :math:`T_{cur}=T_{i}` 时，设置 :math:`\eta_t = \eta_{min}` 。
    当 :math:`T_{cur}=0` （重启后），设置 :math:`\eta_t=\eta_{max}` 。

    请参阅论文 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_ 。

    参数：
        - **base_lr** (float) - 初始学习率。
        - **t_0** (int) - 第一个重启的周期数。
        - **t_mult** (int, 可选) - 重启周期的倍数。默认值： ``1`` 。
        - **eta_min** (float, 可选) - 学习率的最小值。默认值： ``0.`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
