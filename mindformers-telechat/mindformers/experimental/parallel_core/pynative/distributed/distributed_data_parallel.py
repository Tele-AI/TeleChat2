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
""" Distributed data parallel wrapper. """
from contextlib import contextmanager
from mindspore import ops
from mindspore.common import dtype as mstype
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_world_size, \
    get_pipeline_model_parallel_rank, get_data_parallel_group, get_data_modulo_expert_parallel_group
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.distributed.param_and_grad_buffer import ParamAndGradBuffer


__all__ = ['DistributedDataParallel']


class DistributedDataParallel(Module):
    """
    DistributedDataParallel wrapper. DistributedDataParallel allocates contiguous memory buffer for parameters
    and gradients. It also support gradient back-propagation computation and communication. When enable overlapping,
    parameters and gradients will be break up into bucekts which is the unit to conduct all-reduce/reduce-scatter
    communication among data parallel group.

    Args:
        config (TrainingConfig): The TrainingConfig object containing the training related configurations.
        ddp_config (DistributedDataParallelConfig): The DistributedDataParallelConfig object containing the ddp
            related configurations.
        module (Module): The module to be wrapped with DDP.
        disable_bucketing (bool): Disable bucketing, which means all parameters and gradients will be assigned
            to one bucket. Default: False.

    Outputs:
        Model wrapped with DistributedDataParallel.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.dataset as ds
        >>> from mindspore.communication.management import init
        >>> from mindspore.nn import SoftmaxCrossEntropyWithLogits
        >>> from mindspore.mint.optim import AdamW
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel import (ColumnParallelLinear,
        ... RowParallelLinear)
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import (initialize_model_parallel,
        ... get_data_parallel_world_size, get_data_parallel_rank, get_data_parallel_group)
        >>> from mindformers.experimental.parallel_core.pynative.config import (OptimizerConfig, ModelParallelConfig,
        ... TransformerConfig, TrainingConfig)
        >>> from mindformers.experimental.parallel_core.pynative.distributed import (DistributedDataParallel,
        ... DistributedDataParallelConfig)
        >>> from mindformers.experimental.parallel_core.pynative.optimizer.distrib_optimizer import DistributedOptimizer
        >>> from tests.st.test_distri_core.utils import TestData, train
        >>> class TestNet2(nn.Cell):
        ...     def __init__(self, config):
        ...         super(TestNet2, self).__init__()
        ...         hidden_size = config.hidden_size
        ...         self.columnlinear = ColumnParallelLinear(input_size=hidden_size, output_size=hidden_size,
        ...                                                  config=config, init_method=config.init_method,
        ...                                                  bias=config.mlp_has_bias, gather_output=False,
        ...                                                  skip_bias_add=False, bias_init=config.bias_init)
        ...         self.rowlinear = RowParallelLinear(input_size=hidden_size, output_size=hidden_size, config=config,
        ...                                            init_method=config.init_method, bias=config.mlp_has_bias,
        ...                                            input_is_parallel=True, skip_bias_add=False,
        ...                                            bias_init=config.bias_init)
        ...         self.loss = SoftmaxCrossEntropyWithLogits()
        ...     def construct(self, input_, label_):
        ...         output, _ = self.columnlinear(input_)
        ...         output, _ = self.rowlinear(output)
        ...         output = ops.sum(output, dim=-1, keepdim=False)
        ...         output = ops.cast(output, mstype.float32)
        ...         loss = self.loss(output, label_)
        ...         return loss
        ...
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> batch_size = 1
        >>> dataset_size = 6
        >>> seq_length = 8
        >>> hidden_size = 4
        >>> tensor_parallel = 1
        >>> bucket_size = 10
        >>> input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
        >>> label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
        >>> dataset = TestData(input_data=input_data, label_data=label_data)
        >>> dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
        >>>                               num_shards=get_data_parallel_world_size(),
        >>>                               shard_id=get_data_parallel_rank())
        >>> dataset = dataset.batch(batch_size)
        >>> parallel_config = ModelParallelConfig()
        >>> training_config = TrainingConfig(parallel_config=parallel_config)
        >>> optimizer_config = OptimizerConfig(parallel_config=parallel_config)
        >>> model_config = TransformerConfig(vocab_size=40000, num_layers=1, num_attention_heads=1, mlp_has_bias=True,
        >>>                                  gated_linear_unit=False, hidden_size=hidden_size,
        >>>                                  ffn_hidden_size=4*hidden_size, hidden_act='gelu',
        >>>                                  parallel_config=parallel_config, params_dtype='float32',
        >>>                                  compute_dtype='float32')
        >>> ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, use_distributed_optimizer=True,
        >>>     bucket_size=bucket_size, average_in_collective=True, enable_mem_align=True)
        >>> network = TestNet2(config=model_config)
        >>> network_with_ddp = DistributedDataParallel(config=training_config, ddp_config=ddp_config, module=network)
        >>> optimizer = AdamW(params=network_with_ddp.get_parameters(), lr=1.0)
        >>> optimizer = DistributedOptimizer(optimizer=optimizer, config=optimizer_config, grad_scaler=None,
        >>>     init_state_fn=None, per_model_buffers=network_with_ddp.buffers,
        >>>     data_parallel_group=get_data_parallel_group(with_context_parallel=True))
        >>> losses = train(epoch_num=1, dataset=dataset, network=network_with_ddp, optimizer=optimizer)
    """
    def __init__(
            self,
            config,
            ddp_config,
            module,
            disable_bucketing=False,
        ):
        super(DistributedDataParallel, self).__init__(auto_prefix=False)
        self.config = config
        self.ddp_config = ddp_config
        self.module = module
        self.param_to_buffer = {}

        if self.ddp_config.bucket_size is None:
            dp_size = get_data_parallel_world_size()
            # bucket_size elem consumes memory: if use fp32(4B), then one bucket ranges from 4M(dp_size=1) to 160M(max)
            self.ddp_config.bucket_size = max(40000000, 1000000 * dp_size)

        self.bucket_size = self.ddp_config.bucket_size
        if get_pipeline_model_parallel_rank() > 0 or disable_bucketing or not self.ddp_config.overlap_grad_reduce:
            self.bucket_size = None

        dense_params = []
        expert_parallel_params = []
        for _, param in self.module.parameters_and_names():
            if not param.requires_grad:
                continue
            param.grad = None
            param.main_grad = None

            param.grad_accumulated = False

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        if config.calculate_per_token_loss:
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = 1.0
            else:
                data_parallel_world_size = get_data_parallel_world_size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # allocate buffer for common params and expert params
        self.buffers = self.allocate_buffers_for_parameters(
            dense_params,
            group=get_data_parallel_group(with_context_parallel=True),
            gradient_scaling_factor=gradient_scaling_factor,
        )
        self.expert_parallel_buffers = self.allocate_buffers_for_parameters(
            expert_parallel_params,
            group=get_data_modulo_expert_parallel_group(),
            gradient_scaling_factor=expert_gradient_scaling_factor,
        )

        # register hook for bucket grad reduce
        self.register_hook_for_params()

    def allocate_buffers_for_parameters(self, input_params, group, gradient_scaling_factor):
        """ allocate buffers for parameters in different dtype group. """
        param_and_grad_dtype_to_params = {}
        # group all params by parameter's data type and their gradient's data type.
        for param in input_params:
            param_dtype = param.dtype
            grad_dtype = mstype.float32 if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            if (param_dtype, grad_dtype) not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = []
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)].append(param)

        buffers = []
        # allocate buffer for each group separately
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                ParamAndGradBuffer(
                    ddp_config=self.ddp_config,
                    param_dtype=param_dtype,
                    grad_dtype=grad_dtype,
                    params=params,
                    data_parallel_group=group,
                    bucket_size=self.bucket_size,
                    param_to_name=None,
                    gradient_scaling_factor=gradient_scaling_factor,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]

        return buffers

    def issue_grad_reduce(self):
        """ issue grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.issue_grad_reduce()

    def final_grad_reduce(self):
        """ finalize grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.final_grad_reduce()

    def register_hook_for_params(self):
        """ register backward hook for each params. """
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param, self.param_to_buffer))

    def set_input_tensor(self, input_tensor):
        """ set input tensor for model"""
        self.module.set_input_tensor(input_tensor)

    def construct(self, *inputs, **inputs_dict):
        """ construct for DistributedDataParallel. """
        output = self.module(*inputs, **inputs_dict)
        return output

    def zero_grad_buffer(self):
        """ reset buffers for the next train iteration. """
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.grad_accumulated = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()

    def enable_sync(self, enable):
        """ enable grad buffer sync or not. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.sync_enabled = enable

    @contextmanager
    def no_sync(self):
        """ context manager helper function. """
        self.enable_sync(False)
        try:
            yield
        finally:
            self.enable_sync(True)

    def _make_param_hook(
            self,
            param,
            param_to_buffer,
        ):
        """ make closure function as the param hook. """
        def param_hook(grad):
            buffer = param_to_buffer[param]
            if not param.grad_accumulated:
                param.main_grad.add_(grad)
            if self.ddp_config.overlap_grad_reduce:
                buffer.register_grad_ready(param)
            if param.grad is None:
                return ops.Tensor(0, param.dtype)
            return param.grad

        return param_hook
