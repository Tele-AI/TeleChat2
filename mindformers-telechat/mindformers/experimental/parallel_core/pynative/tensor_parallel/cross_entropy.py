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
import mindspore as ms
from mindspore import nn, mint
from mindspore import ops as P
from mindspore.ops import ReduceOp
from mindspore.common.api import _pynative_executor
import mindspore.communication.comm_func as comm_func
from mindformers.experimental.parallel_core.pynative.parallel_state import get_tensor_model_parallel_group, \
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry

__all__ = ['VocabParallelCrossEntropy']


class VocabParallelCrossEntropy(nn.Cell):
    r"""
    Calculate the paralleled cross entropy loss.

    Inputs:
        - **vocab_parallel_logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
          The output logits of the backbone.
        - **target** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.
        - **label_smoothing** (Float) - smoothing factor, must be in range[0.0, 1.0).

    Outputs:
        The corresponding cross entropy loss.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel.cross_entropy import (
        ...     VocabParallelCrossEntropy
        ... )
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import (
        ...     initialize_model_parallel,
        ...     get_tensor_model_parallel_world_size,
        ...     get_data_parallel_world_size
        ... )
        >>> init()
        >>> initialize_model_parallel()
        >>> loss = VocabParallelCrossEntropy()
        >>> logits = Tensor([[2., 1., 0.1]], mstype.float32)
        >>> labels = Tensor([1], mstype.int32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (1,)
        >>> print(output)
        [1.41703]
    """

    # pylint: disable=W0613
    def __init__(self, *args, **kwargs):
        super(VocabParallelCrossEntropy, self).__init__()
        self.label_smoothing = None
        self.vocab_size = None
        self.saved_tensors = [[], [], []]
        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tensor_model_parallel_group()

    def _calculate_logits_max(self, vocab_parallel_logits):
        logits_max = mint.max(vocab_parallel_logits, dim=-1)[0]
        return vocab_parallel_logits, logits_max

    def _get_vocab_range(self, partition_vocab_size, rank):
        vocab_start_index = rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size
        return vocab_start_index, vocab_end_index

    def _calculate_predicted_logits(self, vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index,
                                    partition_vocab_size):
        """ calculate predicted logits """
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        left = target < vocab_start_index
        right = target >= vocab_end_index
        target_mask = left.astype(ms.int32) | right.astype(ms.int32)
        target_mask = target_mask.astype(ms.bool_)
        masked_target = target - vocab_start_index
        masked_target = P.masked_fill(masked_target, target_mask, ms.Tensor(0, masked_target.dtype))

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = mint.arange(start=0, end=logits_2d.shape[0], step=1)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits = P.masked_fill(predicted_logits, target_mask, ms.Tensor(0.0, predicted_logits.dtype))

        exp_logits = mint.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(axis=-1)
        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    def _calculate_cross_entropy_loss(self, exp_logits, predicted_logits, sum_exp_logits):
        loss = mint.log(sum_exp_logits) - predicted_logits
        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1))
        return exp_logits, loss

    def _prepare_gradient_calculation_operands(self, softmax, target_mask):
        grad_input = softmax
        partition_vocab_size = softmax.shape[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = mint.arange(start=0, end=grad_2d.shape[0], step=1)
        target_mask = target_mask.astype(ms.float32)
        softmax_update = 1.0 - target_mask.view(-1)
        return grad_2d, arange_1d, softmax_update, grad_input

    # pylint: disable=W0613, C0111
    def _calculate_gradients(self, grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output,
                             softmax):
        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        grad_2d = grad_2d.reshape(softmax.shape)
        grad_input = grad_2d.mul(grad_output.unsqueeze(dim=-1))
        return grad_input

    def construct(self, vocab_parallel_logits, target, label_smoothing=0.0):
        """ construct method """
        vocab_parallel_logits, logits_max = self._calculate_logits_max(vocab_parallel_logits)
        if self.tp_world_size > 1:
            logits_max = comm_func.all_reduce(logits_max, op=ReduceOp.MAX, group=self.tp_group)[0]

        partition_vocab_size = vocab_parallel_logits.shape[-1]
        vocab_start_index, vocab_end_index = self._get_vocab_range(partition_vocab_size,
                                                                   get_tensor_model_parallel_rank())

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits
        ) = self._calculate_predicted_logits(vocab_parallel_logits, target, logits_max, vocab_start_index,
                                             vocab_end_index, partition_vocab_size)

        if self.tp_world_size > 1:
            predicted_logits = comm_func.all_reduce(predicted_logits, group=self.tp_group)[0]

        if self.tp_world_size > 1:
            sum_exp_logits = comm_func.all_reduce(sum_exp_logits, group=self.tp_group)[0]

        exp_logits, loss = self._calculate_cross_entropy_loss(exp_logits, predicted_logits, sum_exp_logits)
        vocab_size = exp_logits.shape[-1]
        if 0.0 < label_smoothing < 1.0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            log_probs = mint.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs
        elif label_smoothing >= 1.0:
            raise Exception("label_smoothing >= 1.0")

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        if _pynative_executor.grad_flag():
            self.saved_tensors[0].append(exp_logits)
            self.saved_tensors[1].append(target_mask)
            self.saved_tensors[2].append(masked_target_1d)
        return loss

    # pylint: disable=W0613, C0111
    def bprop(self, *args):
        grad_output = args[-1]
        softmax, target_mask, masked_target_1d = self.saved_tensors[0].pop(0),\
            self.saved_tensors[1].pop(0), self.saved_tensors[2].pop(0)

        (
            grad_2d,
            arange_1d,
            softmax_update,
            grad_input
        ) = self._prepare_gradient_calculation_operands(softmax, target_mask)

        if self.label_smoothing > 0.0:
            smoothing = self.label_smoothing * self.vocab_size / (self.vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / self.vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
            grad_2d = grad_2d.reshape(softmax.shape)
            grad_input = grad_2d.mul(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = self._calculate_gradients(grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input,
                                                   grad_output, softmax)
        self.saved_tensors = [[], [], []]

        return grad_input, None, None


ModuleRegistry.register(nn.CrossEntropyLoss, ModuleType.LOSS_FUNC)
ModuleRegistry.register(VocabParallelCrossEntropy, ModuleType.LOSS_FUNC)
