mindformers.core
==================

核心模块，包含运行时上下文、损失函数、优化器、学习率、回调函数和评估指标。

运行时上下文
------------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.build_context
    mindformers.core.get_context
    mindformers.core.init_context
    mindformers.core.set_context

损失函数
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.CrossEntropyLoss

优化器
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.AdamW
    mindformers.core.Came

学习率
----------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.LearningRateWiseLayer
    mindformers.core.ConstantWarmUpLR
    mindformers.core.LinearWithWarmUpLR
    mindformers.core.CosineWithWarmUpLR
    mindformers.core.CosineWithRestartsAndWarmUpLR
    mindformers.core.PolynomialWithWarmUpLR
    mindformers.core.CosineAnnealingLR
    mindformers.core.CosineAnnealingWarmRestarts

回调函数
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.CheckpointMonitor
    mindformers.core.EvalCallBack
    mindformers.core.MFLossMonitor
    mindformers.core.ProfileMonitor
    mindformers.core.SummaryMonitor

评估指标
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:

    mindformers.core.EntityScore
    mindformers.core.EmF1Metric
    mindformers.core.PerplexityMetric
    mindformers.core.PromptAccMetric
