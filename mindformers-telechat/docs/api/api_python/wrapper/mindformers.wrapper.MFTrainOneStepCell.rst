mindformers.wrapper.MFTrainOneStepCell
======================================

.. py:class:: mindformers.wrapper.MFTrainOneStepCell(network, optimizer, use_clip_grad=False, max_grad_norm=1.0, scale_sense=1.0, local_norm=False, **kwargs)

    MindFormers的单步训练包装接口。
    使用损失缩放、梯度裁剪、梯度累积、指数移动平均等进行网络训练。
    这是一个带有损失缩放的训练步骤。它接收一个网络、一个优化器以及一个损失缩放更新的 Cell（或 Tensor）作为参数。损失缩放值可以在主机端或设备端进行更新。如果你想在主机端更新，使用 Tensor 类型的值作为 scale_sense；否则，使用一个 Cell 实例作为 scale_sense 来更新损失缩放。

    参数：
        - **network** (Cell) - 训练网络。网络只支持单输出。
        - **optimizer** (Cell) - 用于更新网络参数的优化器。
        - **use_clip_grad** (bool, 可选) - 是否使用梯度裁剪功能。默认值： ``False`` 。
        - **max_grad_norm** (float, 可选) - 最大梯度范数值。默认值： ``1.0`` 。
        - **scale_sense** (Union[Tensor, Cell], 可选) - 如果该值是一个 Cell，它将被 MFTrainOneStepCell 调用来更新损失缩放。如果该值是一个 Tensor，可以通过 set_sense_scale 修改损失缩放，其形状应为 :math:`()` 或 :math:`(1,)`。
        - **local_norm** (bool, 可选) - 是否计算局部范数。默认值： ``False`` 。
        - **kwargs** (Any) - 其他参数。

    输入：
        - **(\*inputs)** (Tuple(Tensor)) - 形状为 :math:`(N, \ldots)` 的输入张量元组。

    输出：
        5个或7个张量的元组，包括损失值、溢出标志，当前的损失缩放值，优化器学习率，全局梯度norm，局部梯度norm和对应分组size：

        - **loss** (Tensor) -  损失值（标量）。
        - **overflow** (Tensor) -  是否发生溢出（布尔值）。
        - **loss scale** (Tensor) -  损失缩放值，形状为 :math:`()` 或 :math:`(1,)`。
        - **learning rate** (Tensor) -  优化器学习率。
        - **global norm** (Tensor) -  全局梯度norm（标量），用于callback打屏日志，仅当 `use_clip_grad=True` 时计算，否则为None。
        - **local_norm** (Tensor) -  分组梯度norm, 用于callback打屏日志，仅当 `local_norm=True` 时返回。
        - **size** (Tensor) -  每组梯度的size，用于callback打屏日志，仅当 `local_norm=True` 时返回。

    异常：
        - **TypeError** - 如果 `scale_sense` 既不是 Cell 也不是 Tensor。
        - **ValueError** - 如果 `scale_sense` 的形状既不是 `(1,)` 也不是 `()`。
