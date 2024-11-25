mindformers.core.CosineAnnealingLR
==================================

.. py:class:: mindformers.core.CosineAnnealingLR(base_lr: float, t_max: int, eta_min: float = 0., **kwargs)

    该方法在 `SGDR: Stochastic Gradient Descent with Warm Restarts` 中提出。注意，这里仅实现了SGDR的余弦退火部分，而不包括重启部分。

    请参阅论文 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_ 。

    使用余弦退火调度设置每个参数组的学习率，其中 :math:`\eta_{max}` 被设为初始学习率， :math:`T_{cur}` 表示自上次在SGDR中重启以来的epoch数量：

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    当 `last_epoch=-1` 时，初始学习率设置为 `lr` 。请注意，由于调度器是递归定义的，学习率可以同时通过其他操作符在此调度器之外进行修改。如果学习率仅由此调度器设置，则每一步的学习率变为：

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    参数：
        - **base_lr** (float) - 初始学习率。
        - **t_max** (int) - 重启周期的最大周期数。
        - **eta_min** (float) - 学习率的最小值。默认值： ``0`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
