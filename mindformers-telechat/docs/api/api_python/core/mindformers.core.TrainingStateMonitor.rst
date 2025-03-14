mindformers.core.TrainingStateMonitor
=====================================

.. py:class:: mindformers.core.TrainingStateMonitor(origin_epochs, config=None, per_print_times=1, dataset_size=None, initial_epoch=0, initial_step=0, global_batch_size=0)

    监控训练过程中指标变化的回调函数。

    参数：
        - **origin_epochs** (int) - 必选。训练的epoch数量。
        - **config** (dict, 可选) - 指标落盘的配置信息字典。字典的键取值如下。默认值： ``None`` ，此时将按照默认的键取值设置。

          - ``"target"`` - 指定要监控的参数的命名或正则表达式。必须是字符串列表，例如["layers.[01]", "attention"]。默认值： ``[".*"]`` ，即选择所有参数。
          - ``"invert"`` - 反选target指定的参数，即target指定的参数不会被监控。默认值： ``False`` 。
          - ``"local_norm_format"`` - 决定local norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。只有指定的参数会被监控，选择 'log' 时可能引入大量打印信息。设置为 ``None`` 以忽略该指标。默认值：``None`` 。
          - ``"device_local_norm_format"`` - 决定device local norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值：``None`` 。
          - ``"local_loss_format"`` - 决定local loss的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值：``None`` 。
          - ``"optimizer_state_format"`` - 决定优化器状态的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。只有指定参数的优化器状态会被监控，选择 'log' 时可能引入大量打印信息。设置为 ``None`` 以忽略该指标。默认值：'tensorboard' 。
          - ``"weight_state_format"`` - 决定权重L2-norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值：'tensorboard' 。
          - ``"throughput_baseline"`` - 模型吞吐量的基线，用于计算线性度。必须为正数。会同时写入日志文件和tensorboard。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"print_struct"`` - 是否打印模型结构。若是，则会在第一个step打印所有可训练参数的名字，并退出训练。默认值： ``False`` 。

        - **per_print_times** (int, 可选) - 每多少次step打印日志信息。默认值： ``1`` 。
        - **dataset_size** (int, 可选) - 数据下沉模式必选。训练的数据集数量。默认值： ``None`` 。
        - **initial_epoch** (int, 可选) - 训练开始的epoch数。默认值： ``0`` 。
        - **initial_step** (int, 可选) - 训练开始的step数。默认值： ``0`` 。
        - **global_batch_size** (int, 可选) - 总BatchSize大小。默认值： ``0`` 。