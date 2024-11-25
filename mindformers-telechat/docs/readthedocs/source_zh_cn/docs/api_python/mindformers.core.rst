mindformers.core
==================

.. automodule:: mindformers.core

mindformers.core
-----------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.build_context
    mindformers.core.init_context

mindformers.core.callback
--------------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.callback.CheckpointMonitor
    mindformers.core.callback.MFLossMonitor
    mindformers.core.callback.ObsMonitor
    mindformers.core.callback.SummaryMonitor
    mindformers.core.callback.ProfileMonitor
    mindformers.core.callback.EvalCallBack

mindformers.core.loss
--------------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.loss.CrossEntropyLoss
    mindformers.core.loss.L1Loss
    mindformers.core.loss.MSELoss
    mindformers.core.loss.SoftTargetCrossEntropy

mindformers.core.lr
--------------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.lr.ConstantWarmUpLR
    mindformers.core.lr.CosineWithRestartsAndWarmUpLR
    mindformers.core.lr.CosineWithWarmUpLR
    mindformers.core.lr.LinearWithWarmUpLR
    mindformers.core.lr.PolynomialWithWarmUpLR
    mindformers.core.lr.CosineAnnealingLR
    mindformers.core.lr.CosineAnnealingWarmRestarts

mindformers.core.metric
--------------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.metric.EntityScore
    mindformers.core.metric.SQuADMetric
    mindformers.core.metric.PerplexityMetric
    mindformers.core.metric.ADGENMetric

mindformers.core.optim
--------------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.optim.FP32StateAdamWeightDecay
