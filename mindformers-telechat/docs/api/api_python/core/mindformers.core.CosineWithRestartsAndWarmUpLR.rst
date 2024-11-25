mindformers.core.CosineWithRestartsAndWarmUpLR
==============================================

.. py:class:: mindformers.core.CosineWithRestartsAndWarmUpLR(learning_rate: float, warmup_steps: int = None, total_steps: int = None, num_cycles: float = 1., lr_end: float = 0., warmup_lr_init: float=0., warmup_ratio: float = None, decay_steps: int = None, **kwargs)

    余弦重启与预热学习率。

    使用余弦重启与预热调度设置每个参数组的学习率，其中 :math:`\eta_{max}` 被设为初始学习率， :math:`T_{cur}` 表示自上次重启以来的步数：

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1
            + \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right), & T_{cur} \neq (2k+1)T_{i}; \ \eta_{t+1}
            & = \eta_{\text{max}}, & T_{cur} = (2k+1)T_{i}.
        \end{aligned}

    当 last_epoch=-1 时，初始学习率设置为 `lr` 。在重启阶段，学习率从最大值重新开始，最终逐渐减小到设定的最小值。这种策略有助于在训练过程中避免陷入局部最优解并加速收敛。

    该方法在 SGDR: Stochastic Gradient Descent with Warm Restarts 中提出，扩展了余弦退火的概念以实现多次重启。

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **warmup_steps** (int) - 预热步骤数。默认值： ``None`` 。
        - **total_steps** (int) - 总步数。默认值： ``None`` 。
        - **num_cycles** (float) - 余弦调度中的波形数量（默认为仅遵循半个余弦从最大值下降到 0）。默认值： ``1.`` 。
        - **lr_end** (float) - 学习率的最终值。默认值： ``0.`` 。
        - **warmup_lr_init** (float) - 预热步骤中的初始学习率。默认值： ``0.`` 。
        - **warmup_ratio** (float) - 预热所用的总训练步骤的比例。默认值： ``None`` 。
        - **decay_steps** (int) - 衰减步骤的数量。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
