mindformers.core
==================

core module, including Runtime Context, Loss, Optimizer, Learning Rate, Callback, and Evaluation Metrics.

Runtime Context
-----------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.build_context
    mindformers.core.get_context
    mindformers.core.init_context
    mindformers.core.set_context

Loss
-----

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.CrossEntropyLoss

Optimizer
----------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.AdamW
    mindformers.core.Came

Learning Rate
--------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.ConstantWarmUpLR
    mindformers.core.CosineAnnealingLR
    mindformers.core.CosineAnnealingWarmRestarts
    mindformers.core.CosineWithRestartsAndWarmUpLR
    mindformers.core.CosineWithWarmUpLR
    mindformers.core.LearningRateWiseLayer
    mindformers.core.LinearWithWarmUpLR
    mindformers.core.PolynomialWithWarmUpLR

Callback
--------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.CheckpointMonitor
    mindformers.core.EvalCallBack
    mindformers.core.MFLossMonitor
    mindformers.core.ProfileMonitor
    mindformers.core.SummaryMonitor

Evaluation Metric
-------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.EntityScore
    mindformers.core.EmF1Metric
    mindformers.core.PerplexityMetric
    mindformers.core.PromptAccMetric