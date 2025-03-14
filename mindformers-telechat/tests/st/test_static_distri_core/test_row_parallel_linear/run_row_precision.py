# pylint: skip-file
import os
import numpy as np

import mindspore as ms
from mindspore import init, value_and_grad
from mindspore.ops.auto_generate import Mul, SumExt

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.tensor_parallel.layers import RowParallelLinear
from mindformers.experimental.utils import init_method_normal


ms.context.set_context(deterministic="ON")
ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

tensor_model_parallel_size = 2
input_size_coeff = 2
output_size_coeff = 2
batch_size = 2
input_size = input_size_coeff * tensor_model_parallel_size
output_size = output_size_coeff * tensor_model_parallel_size

config_ = TransformerConfig()
config_.data_parallel = 1
config_.tensor_parallel = tensor_model_parallel_size
config_.context_parallel = 1


net = RowParallelLinear(input_size=input_size, output_size=output_size,
                        config=config_, compute_dtype=ms.dtype.float16,
                        init_method=init_method_normal(), transpose_b=True, bias=True)
net = net.to_float(ms.float16)
rank = ms.communication.get_rank()

loss_weight = np.array([[[0.3044, 0.907, 0.7637, 0.6104],
                         [0.8916, 0.909, 0.665, 0.6787]],
                        [[0.3044, 0.907, 0.7637, 0.6104],
                         [0.8916, 0.909, 0.665, 0.6787]]], dtype=np.float16)

input_tensor = np.array([[[[0.2268, 0.503],
                           [0.07837, 0.07056]],
                          [[0.983, 0.697],
                           [0.5645, 0.1914]]],
                         [[[0.2151, 0.799],
                           [0.297, 0.06915]],
                          [[0.6904, 0.4407],
                           [0.3562, 0.5625]]]
                        ], dtype=np.float16)

output_tensor = np.array([[[[-0.006462, -0.01715, -0.01213, 0.004517],
                            [0.006588, 0.002174, 0.01022, -0.001186]],
                           [[-0.00705, 0.0002718, 0.00444, 0.000682],
                            [-0.00813, -0.006413, 0.007477, 0.00915]]],
                          [[[-0.006462, -0.01715, -0.01213, 0.004517],
                            [0.006588, 0.002174, 0.01022, -0.001186]],
                           [[-0.00705, 0.0002718, 0.00444, 0.000682],
                            [-0.00813, -0.006413, 0.007477, 0.00915]]]
                        ], dtype=np.float16)

weight = np.array([[[-0.02452, -0.0001863],
                    [0.00559, -0.00522],
                    [0.002342, -0.03772],
                    [0.01072, -0.01779]]
                   [[0.03084, -0.009315],
                    [0.01246, -0.02312],
                    [0.04364, -0.003859],
                    [-0.0062, 0.01548]]], dtype=np.float16)


bias_grad = np.array([[2.393, 3.633, 2.857, 2.578],
                      [2.393, 3.633, 2.857, 2.578]], dtype=np.float16)

bias = np.array([[0., 0., 0., 0.],
                 [0., 0., 0., 0.]], dtype=np.float16)

weight_grad = np.array([[[1.883, 1.197],
                         [3.363, 2.652],
                         [2.703, 2.182],
                         [2.35, 1.82]]
                        [[1.717, 1.881],
                         [2.83, 3.398],
                         [2.252, 2.732],
                         [1.992, 2.371]
                         ]], dtype=np.float16)

input_grad = np.array([[[[0.01187, -0.0889],
                         [-0.0159, -0.08417]],
                        [[0.01187, -0.0889],
                         [-0.0159, -0.08417]]],
                       [[[0.10046, -0.0346],
                         [0.1273, -0.04276]],
                        [[0.10046, -0.0346],
                         [0.1273, -0.04276]]]], dtype=np.float16)

inputs = ms.Tensor(input_tensor, ms.float16)
output_pta = ms.Tensor(output_tensor, ms.float16)
weight = ms.Tensor(weight, ms.float16)
bias = ms.Tensor(bias, ms.float16)
loss_weight = ms.Tensor(loss_weight, ms.float16)
bias_grad_pta = ms.Tensor(bias_grad, ms.float16)
weight_grad_pta = ms.Tensor(weight_grad, ms.float16)
input_grad_pta = ms.Tensor(input_grad, ms.float16)


inputs = ms.ops.concat(inputs.chunk(inputs.shape[0], 0), -1).squeeze(0)
output_pta = output_pta[0]
weight = ms.ops.concat(weight.chunk(weight.shape[0], 0), -1).squeeze(0)
bias = bias[0]
loss_weight = loss_weight[0]
bias_grad_pta = bias_grad_pta[rank]
weight_grad_pta = weight_grad_pta[rank]
input_grad_pta = input_grad_pta[rank]

net.weight = ms.Parameter(weight)
net.weight = ms.Parameter(weight)

mul = Mul().shard(((1, 1), (1, 1, 1)))
sum_pri = SumExt().shard(((1, 1, 1),))

def forward(inputs, loss_weight):
    output_, _ = net(inputs)
    loss_ = sum_pri(mul(loss_weight, output_))
    return loss_, output_

grad_fn = value_and_grad(forward, grad_position=0, weights=net.trainable_params(), has_aux=True)
(loss, output), (input_grad, params_grad) = grad_fn(inputs, loss_weight)
weight_grad, bias_grad = params_grad

input_grad = input_grad[:, :, rank*input_size_coeff:(rank+1)*input_size_coeff]

assert bias_grad.equal(bias_grad_pta*tensor_model_parallel_size).all()
assert weight_grad.equal(weight_grad_pta).all()
assert input_grad.equal(input_grad_pta).all()
assert output.equal(output_pta).all()
