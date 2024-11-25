# Copyright 2024 Huawei Technologies Co., Ltd
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
"""loss function"""
from mindspore import nn, mint

from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry

__all__ = ['get_loss_func']


class LossWithMask(nn.Cell):
    """
    Calculate the loss with mask and mean reduction.

    Args:
        - **loss_func** (Function) - Loss function.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        The corresponding cross entropy loss.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, nn
        >>> loss = LossWithMask(nn.CrossEntropyLoss())
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    # pylint: disable=W0613
    def __init__(self, loss_func, *args, **kwargs):
        super(LossWithMask, self).__init__()
        self.loss_func = loss_func

    def construct(self, logits, label, input_mask):
        loss_reduce = self.loss_func(logits, label)
        input_mask = input_mask.view(-1)
        loss = mint.sum(loss_reduce * input_mask) / input_mask.sum()
        return loss


def get_loss_func(training_config, return_instance: bool = True, **kwargs):
    """
    Get the loss function based on the provided loss function configuration.

    Args:
        training_config (TrainingConfig): The configuration object for training.

    Returns:
        loss_fn (callable): The loss function based on the provided configuration.

    Raises:
        ValueError: If the specified loss function type is not supported.
    """
    loss_func_kwargs = training_config.loss_func_kwargs
    loss_func_kwargs["reduction"] = training_config.loss_reduction
    loss_func_type = loss_func_kwargs['loss_func_type']
    if "CrossEntropyLoss" in loss_func_type:
        loss_func_kwargs["reduction"] = 'none'
    loss_func_cls = ModuleRegistry.get_item(module_type=ModuleType.LOSS_FUNC, item_name=loss_func_type)
    if return_instance:
        loss_func_kwargs.update(kwargs)
        loss_func_kwargs = ModuleRegistry.get_needed_params_for_init(loss_func_cls, loss_func_kwargs)
        return LossWithMask(loss_func=loss_func_cls(**loss_func_kwargs))
    return loss_func_cls
