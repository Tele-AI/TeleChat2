mindformers.core.Came
=====================

.. py:class:: mindformers.core.Came(params, learning_rate=None, eps=(1e-30, 1e-3, 1e-16), clip_threshold=1.0, decay_rate=0.8, beta1=0.9, beta3=0.99, weight_decay=0.0, scale_parameter=False, relative_step=False, warmup_init=False, compression=False, loss_scale=1)

    通过Confidence-guided Adaptive Memory Efficient Optimization (Came)算法更新梯度。

    请参阅论文 `CAME: Confidence-guided Adaptive Memory Efficient Optimization <https://arxiv.org/abs/2307.02047>`_ 。

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 当 `params` 是将要更新的 `Parameter` 列表时，`params` 中的元素必须是 `Parameter` 类。
        - **learning_rate** (Union[float, Tensor]) - 学习率。当 `learning_rate` 是一维 `Tensor` 时。如果 `learning_rate` 是 `int` 类型，将被转换为 `float`。默认值： ``None`` 。
        - **eps** (Union[list, tuple]) - 分别为平方梯度、参数比例和不稳定矩阵的正则化常数。默认值： ``(1e-30, 1e-3, 1e-16)`` 。
        - **clip_threshold** (float) - 最终梯度更新的均方根阈值。默认值： ``1.0`` 。
        - **decay_rate** (float) - 用于计算平方梯度的运行平均值的系数。应在 [0.0, 1.0] 范围内。默认值： ``0.8`` 。
        - **beta1** (float) - 计算梯度运行平均值的系数。应在 [0.0, 1.0] 范围内。默认值： ``0.9`` 。
        - **beta3** (float) - 计算梯度运行平均值的系数。应在 [0.0, 1.0] 范围内。默认值： ``0.99`` 。
        - **weight_decay** (float) - 权重衰减 (L2 惩罚)。必须大于或等于 0。应在 [0.0, 1.0] 范围内。默认值： ``0.0`` 。
        - **scale_parameter** (bool) - 如果为 ``True``，则学习率按参数的均方根缩放。默认值： ``False`` 。
        - **relative_step** (bool) - 如果为 ``True``，则计算时间相关学习率，而不是外部学习率。默认值： ``False`` 。
        - **warmup_init** (bool) - 时间相关学习率计算取决于是否使用预热初始化。默认值： ``False`` 。
        - **compression** (bool) - 如果为 ``True``，则运行平均指数的数据类型将压缩为 float16。默认值： ``False`` 。
        - **loss_scale** (int) - 损失缩放的整数值。应大于 0。通常使用默认值。仅当使用 `FixedLossScaleManager` 进行训练且 `FixedLossScaleManager` 中的 `drop_overflow_update` 设置为 ``False`` 时，此值需要与 `FixedLossScaleManager` 中的 `loss_scale` 相同。有关更多详细信息，请参阅类 `mindspore.amp.FixedLossScaleManager`。默认值： ``1`` 。

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，形状与 `params` 相同。

    输出：
        Tensor[bool]，值为 `True` 。

    异常：
        - **TypeError** - 如果 `learning_rate` 不是 int、float、Tensor、Iterable 或 LearningRateSchedule 中的一种。
        - **TypeError** - 如果 `parameters` 的元素既不是 Parameter 也不是 dict。
        - **TypeError** - 如果 `decay_rate`、`weight_decay`、`beta1`、`beta3`、`eps` 或 `loss_scale` 不是 float。
        - **TypeError** - 如果 `use_locking` 或 `use_nesterov` 不是 bool。
        - **ValueError** - 如果 `loss_scale` 或 `eps` 小于或等于 0。
        - **ValueError** - 如果 `decay_rate`、`weight_decay`、`beta1` 或 `beta3` 不在 [0.0, 1.0] 范围内。
