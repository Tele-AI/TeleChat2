# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Self-Define Wrapper."""
import hashlib
from copy import deepcopy

from mindspore import nn, Parameter, ParallelMode
from mindspore.common import RowTensor
from mindspore.common.tensor import Tensor
from mindspore.communication.management import (create_group, get_rank, get_group_size)
from mindspore.nn.wrap.cell_wrapper import _MicroBatch
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._utils import _get_enable_parallel_optimizer
import mindspore.common.dtype as mstype

from mindformers.core.clip_grad import ClipGradNorm
from mindformers.core.optim import FusedCastAdamWeightDecay
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.version_control import get_identity

__all__ = ['MFTrainOneStepCell', 'MFPipelineWithLossScaleCell', 'PipelineCellWithTwoOutput',
           'GradAccumulationCellWithTwoOutput']

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return F.cast(grad, mstype.float32) * reciprocal(scale)


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)

get_square_sum = C.MultitypeFuncGraph("get_square_sum")
get_size = C.MultitypeFuncGraph("get_size")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(F.cast(grad, mstype.float32)), ()) * value
    norm = F.expand_dims(norm, 0)
    return norm


# pylint: disable=E0102
@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = P.ReduceSum(False)(F.square(F.cast(grad, mstype.float32)), ())
    norm = F.expand_dims(norm, 0)
    return norm


@get_size.register("Tensor")
def _get_size(grad):
    size = P.Size()(grad)
    return size


class LocalNorm(nn.Cell):
    def __init__(self):
        super(LocalNorm, self).__init__()
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads)
        size = self.hyper_map(get_size, grads)
        return square_sum, size


# pylint: disable=W1401
@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStep For MindFormer.
    Network training with loss scaling, grad clip, gradient accumulation, exponential moving average and so on.

    This is a training step with loss scaling. It takes a network, an optimizer and a scale update Cell(or a Tensor) as
    args. The loss scale value can be updated in both host side or device side. If you want to update it on
    host side, using a value of Tensor type as `scale_sense`, otherwise, using a Cell instance for updating loss
    scale as `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the network parameters.
        use_clip_grad (bool, optional): Whether to use the gradient clipping function. Default: ``False`` .
        max_grad_norm (float, optional): Maximum gradient value. Default: ``1.0`` .
        scale_sense (Union[Tensor, Cell], optional): If this value is a Cell, it will be called by `MFTrainOneStepCell`
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        local_norm (bool, optional): Whether to calculate the local norm. Default: ``False`` .
        kwargs (Any): Additional parameters.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 5 or 7 Tensor, the loss, overflow flag, current loss scale value, learning rate,
        global grads norm, local grads norm and size of local norm grads.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.
        - **learning rate** (Tensor) -  A scalar, the learning rate of the optimizer.
        - **global norm** (Tensor) -  A scalar, the global norm of all grads, only be calculated
          when `use_clip_grad=True`, otherwise None.
        - **local_norm** (Tensor) -  The local norm of the grads by group, only be returned when `local_norm=True`.
        - **size** (Tensor) -  The sizes of each grads group, only be returned when `local_norm=True`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Examples:
        >>> from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
        >>> from mindformers.wrapper import MFTrainOneStepCell
        >>> import mindspore as ms
        >>> from mindformers.core.optim import AdamW
        >>> import numpy as np
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>>
        >>> config = LlamaConfig(num_layers=2)
        >>> net = LlamaForCausalLM(config=config)
        >>> net.set_train(True)
        >>> optimizer = AdamW(net.trainable_params())
        >>>
        >>> mft = MFTrainOneStepCell(net, optimizer)
        >>> inputs = ms.Tensor(np.ones([1, 2049]), ms.int32)
        >>> out = mft(inputs)
        >>>
        >>> loss, overflow, loss_scale, lr, global_norm = out
        >>> print(loss.shape, overflow, loss_scale, lr, global_norm)
        (1,) False 1.0 0.001, None
    """

    def __init__(self,
                 network,
                 optimizer,
                 use_clip_grad=False,
                 max_grad_norm=1.0,
                 scale_sense=1.0,
                 local_norm=False,
                 **kwargs):
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense)
        super(MFTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.use_clip_grad = use_clip_grad
        if isinstance(optimizer, FusedCastAdamWeightDecay):
            self.use_grad_norm = True
        else:
            self.use_grad_norm = False
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.parallel_config = kwargs.pop("parallel_config", None)
        self.learning_rate = deepcopy(self.optimizer.learning_rate)
        self.localnorm = LocalNorm()
        self.concat = P.Concat()
        self.local_norm = local_norm

    def construct(self, *inputs):
        """forward and backward."""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)

        if self.local_norm:
            local_norm, size = self.localnorm(grads)
            local_norm = self.concat(local_norm)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        global_norm = None
        if self.use_clip_grad:
            grads, global_norm = self.clip_grad_norm(grads)

        learning_rate = self.learning_rate
        if self.optimizer.dynamic_lr:
            if self.optimizer.is_group_lr:
                learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
            else:
                learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())

        # if there is no overflow, do optimize
        if not overflow:
            if self.use_clip_grad and self.use_grad_norm:
                loss = F.depend(loss, self.optimizer(grads, global_norm))
            else:
                loss = F.depend(loss, self.optimizer(grads))
        if self.local_norm:
            return loss, overflow, scaling_sens, learning_rate, global_norm, local_norm, size
        return loss, overflow, scaling_sens, learning_rate, global_norm


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")


@grad_scale.register("Tensor", "Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad_scale_factor, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    grad_val = F.cast(F.equal(accu_grad, accu_grad), F.dtype(accu_grad))
    zeros = F.mul(grad_val, 0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    new_grad = new_grad * reciprocal(grad_scale_factor)
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad_scale_factor, grad, accu_grad):
    new_grad = grad * F.cast(reciprocal(scale), F.dtype(grad))
    accu_grad = F.depend(accu_grad, new_grad)
    grad_val = F.cast(F.equal(accu_grad, accu_grad), F.dtype(accu_grad))
    zeros = F.mul(grad_val, 0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    new_grad = new_grad * reciprocal(grad_scale_factor)
    return new_grad


class GradAccumulationCellWithTwoOutput(nn.Cell):
    """
        Wrap the network with Micro Batch to enable the grad accumulation in semi_auto_parallel/auto_parallel mode.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self, network, micro_size):
        super(GradAccumulationCellWithTwoOutput, self).__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        if not isinstance(network, nn.Cell):
            raise TypeError("For 'GradAccumulationCellWithTwoOutput', the argument 'network' must cell type, "
                            "but got the type : {}.".format(type(network)))
        if not isinstance(micro_size, int):
            raise TypeError("For 'GradAccumulationCellWithTwoOutput', the argument 'micro_size' must be integer, "
                            "but got the type : {}.".format(type(micro_size)))
        if micro_size <= 0:
            raise ValueError("For 'GradAccumulationCellWithTwoOutput', the argument 'micro_size' must be large than 0, "
                             "but got {}.".format(micro_size))
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            micro_input.strided_slice.add_prim_attr("grad_accu_num", micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("forward_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """Construct function for pipeline with multiple outputs."""
        ret = None
        ret2 = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output1, output2 = self.network(*micro_input)

            if ret is not None:
                ret = self.add_list[i](ret, output1)
            else:
                ret = output1

            if ret2 is not None:
                ret2 = self.add_list[i](ret2, output2)
            else:
                ret2 = output2

        loss = ret, ret2
        return loss


def _get_pipeline_group():
    """
    Calculate the communication group between all pipeline stages
    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class PipelineCellWithTwoOutput(nn.Cell):
    """
        Slice MiniBatch into finer-grained MicroBatch for use in pipeline-parallel training.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """
    def __init__(self, network, micro_size):
        super(PipelineCellWithTwoOutput, self).__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        if not isinstance(network, nn.Cell):
            raise TypeError("For 'PipelineCellWithTwoOutput', the argument 'network' must cell type, "
                            "but got the type : {}.".format(type(network)))
        if not isinstance(micro_size, int):
            raise TypeError("For 'PipelineCellWithTwoOutput', the argument 'micro_size' must be integer, "
                            "but got the type : {}.".format(type(micro_size)))
        if micro_size <= 0:
            raise ValueError("For 'PipelineCellWithTwoOutput', the argument 'micro_size' must be large than 0, "
                             "but got {}.".format(micro_size))
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("pipeline_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """
        Construct function for pipeline with multiple input
        Args:
            *inputs:

        Returns:

        """
        ret = None
        ret2 = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output1, output2 = self.network(*micro_input)

            if ret is not None:
                ret = self.add_list[i](ret, output1)
            else:
                ret = output1

            if ret2 is not None:
                ret2 = self.add_list[i](ret2, output2)
            else:
                ret2 = output2

        loss = ret, ret2
        return loss


# pylint: disable=W1401
@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFPipelineWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Append a train-one-step cell with loss scale of pipeline parallel for MindFormers.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        use_clip_grad (bool, optional): Whether to use gradient clipping. Default: ``True`` .
        max_grad_norm (float, optional): Maximum gradient constraint value. Default: ``1.0`` .
        scale_sense (Union[Tensor, Cell], optional): Cell to do the loss scale. Default: ``1.0`` .
        micro_batch_num (int, optional): Micro batch number of pipeline parallel. Default: ``1`` .
        local_norm (bool, optional): Whether to calculate the local norm. Default: ``False`` .
        kwargs (Any): Additional parameters.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 5 or 7 Tensor, the loss, overflow flag, current loss scale value, learning rate,
        global grads norm, local grads norm and size of local norm grads.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.
        - **learning rate** (Tensor) -  A scalar, the learning rate of the optimizer.
        - **global norm** (Tensor) -  A scalar, the global norm of all grads, only be calculated
          when `use_clip_grad=True`, otherwise None.
        - **local_norm** (Tensor) -  The local norm of the grads by group, only be returned when `local_norm=True`.
        - **size** (Tensor) -  The sizes of each grads group, only be returned when `local_norm=True`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().
        ValueError: If the parallel mode is not one of [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL].
    """

    def __init__(self, network, optimizer, use_clip_grad=True, max_grad_norm=1.0,
                 scale_sense=1.0, micro_batch_num=1, local_norm=False, calculate_per_token_loss=False, **kwargs):
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense)
        super(MFPipelineWithLossScaleCell, self).__init__(network, optimizer, scale_sense)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = get_identity()
        self.degree = 1
        self.cast = P.Cast()
        self.status = Tensor([0] * 8, mstype.int32)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        if self.parallel_mode not in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]:
            raise ValueError(f"ParallelMode must be one of "
                             f"[ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL], but found "
                             f"{self.parallel_mode}.")
        self.allreduce = P.AllReduce()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.reshape = P.Reshape()
        self.loss_scaling_manager = None
        if isinstance(scale_sense, nn.Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of 'scale_sense' must be (1,) or (), but got {}"
                                 .format(scale_sense.shape))
        else:
            raise TypeError("The 'scale_sense' must be Cell or Tensor, but got {}".format(type(scale_sense)))
        self.opt_shard = _get_enable_parallel_optimizer()
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.micro_size = micro_batch_num
        self.parallel_config = kwargs.pop("parallel_config", None)
        self.learning_rate = deepcopy(self.optimizer.learning_rate)
        self.localnorm = LocalNorm()
        self.concat = P.Concat()
        self.local_norm = local_norm
        self.zero_t = Tensor([0], dtype=mstype.float32)
        # create allreduce for synchronize denominator
        pipeline_group_list, pipeline_group_name = _get_pipeline_group()
        hashed = hashlib.md5(pipeline_group_name.encode()).hexdigest()[:48]
        pipeline_group_name = str(hashed)
        create_group(pipeline_group_name, pipeline_group_list)
        self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        self.calculate_per_token_loss = calculate_per_token_loss
        self.grad_scale_factor = Tensor([1], dtype=mstype.float32)
        self.zero_t = Tensor([0], dtype=mstype.float32)

    @C.add_flags(has_effect=True)
    def construct(self, *inputs):
        """The construct processes of pipeline wrapper cell."""
        scaling_sens = self.scale_sense
        if self.calculate_per_token_loss:
            numerator, denominator = self.network(*inputs)
            denominator = self.allreduce2(denominator)
            loss = numerator / denominator
            scaling_sens_filled = C.ones_like(numerator) * F.cast(scaling_sens, F.dtype(numerator))
            scaling_sens_filled2 = self.zero_t * F.cast(scaling_sens, F.dtype(denominator))

            grads = self.grad(self.network, self.weights)(*inputs,
                                                          (self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled2, mstype.float32)))
            grad_scale_factor = denominator
        else:
            loss = self.network(*inputs)
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, self.weights)(*inputs,
                                                          self.cast(scaling_sens_filled / self.micro_size,
                                                                    mstype.float32))
            grad_scale_factor = self.grad_scale_factor

        if self.local_norm:
            local_norm, size = self.localnorm(grads)
            local_norm = self.concat(local_norm)

        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree, grad_scale_factor),
                                   grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree, grad_scale_factor), grads,
                                   accu_grads)

        global_norm = None
        if self.use_clip_grad:
            grads, global_norm = self.clip_grad_norm(grads)

        learning_rate = self.learning_rate
        if self.optimizer.dynamic_lr:
            if self.optimizer.is_group_lr:
                learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
            else:
                learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())

        # sum overflow flag over devices
        cond = self.get_overflow_status(self.status, grads)
        cond = F.depend(cond, grads)
        overflow = self.process_loss_scale(cond)

        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))

        if self.local_norm:
            return loss, overflow, scaling_sens.value(), learning_rate, global_norm, local_norm, size
        return loss, overflow, scaling_sens.value(), learning_rate, global_norm
