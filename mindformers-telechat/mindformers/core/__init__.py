# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindFormers Core."""
from .lr import (
    ConstantWarmUpLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CosineWithRestartsAndWarmUpLR,
    CosineWithWarmUpLR,
    LearningRateWiseLayer,
    LinearWithWarmUpLR,
    PolynomialWithWarmUpLR,
    build_lr
)
from .loss import (
    CompareLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    SoftTargetCrossEntropy,
    build_loss
)
from .optim import (
    AdamW,
    Came,
    FP32StateAdamWeightDecay,
    FusedAdamWeightDecay,
    FusedCastAdamWeightDecay,
    build_optim
)
from .metric import (
    ADGENMetric,
    EmF1Metric,
    EntityScore,
    PerplexityMetric,
    PromptAccMetric,
    SQuADMetric,
    build_metric
)
from .callback import (
    CheckpointMonitor,
    EvalCallBack,
    MFLossMonitor,
    ObsMonitor,
    ProfileMonitor,
    SummaryMonitor,
    build_callback
)
from .context import (
    build_context,
    get_context,
    init_context,
    set_context
)
from .clip_grad import ClipGradNorm
from .parallel_config import (
    build_parallel_config,
    reset_parallel_config
)

__all__ = []
__all__.extend(callback.__all__)
__all__.extend(context.__all__)
__all__.extend(loss.__all__)
__all__.extend(lr.__all__)
__all__.extend(metric.__all__)
__all__.extend(optim.__all__)
