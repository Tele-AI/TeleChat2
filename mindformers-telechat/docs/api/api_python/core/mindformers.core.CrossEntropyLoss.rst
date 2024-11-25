mindformers.core.CrossEntropyLoss
=================================

.. py:class:: mindformers.core.CrossEntropyLoss(parallel_config=default_dpmp_config, check_for_nan_in_loss_and_grad=False, calculate_per_token_loss, **kwargs)

    计算预测值和目标值之间的交叉熵损失。

    CrossEntropyLoss支持两种不同的目标值(target):

    - 类别索引 (int)，取值范围为 :math:`[0, C)` 其中 :math:`C` 为类别数，当reduction为 ``'none'`` 时，交叉熵损失公式如下：

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重，N表示batch size， :math:`c` 限定范围为[0, C-1]，表示类索引，其中 :math:`C` 表示类的数量。

      若reduction不为 ``'none'`` （默认为 ``'mean'`` ），则

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    - 类别概率 (float)，用于目标值为多个类别标签的情况。当reduction为 ``'none'`` 时，交叉熵损失公式如下：

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重，N表示batch size， :math:`c` 限定范围为[0, C-1]，表示类索引，其中 :math:`C` 表示类的数量。

      若reduction不为 ``'none'`` （默认为 ``'mean'`` ），则

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    参数：
        - **parallel_config** (:class:`mindformers.modules.transformer.op_parallel_config.OpParallelConfig`) - 并行配置字典，用于控制并行训练的策略。默认值： ``default_dpmp_config`` 。
        - **check_for_nan_in_loss_and_grad** (bool) - 是否打印局部损失。默认值： ``False`` 。
        - **calculate_per_token_loss** (bool) - 是否计算每个token的损失。默认值： ``False`` 。

    输入：
        - **logits** (Tensor) - 输入预测值，shape为 :math:`(N, C)` 。输入值需为对数概率。数据类型仅支持float32或float16。
        - **label** (Tensor) - 输入目标值。shape为 :math:`(N,)` 。
        - **input_mask** (Tensor) - 损失掩码，shape为 :math:`(N,)` 。用于指定需要计算损失的位置。若值为0，则对应位置不计算损失。

    返回：
        Tensor，一个数据类型与logits相同的Tensor。