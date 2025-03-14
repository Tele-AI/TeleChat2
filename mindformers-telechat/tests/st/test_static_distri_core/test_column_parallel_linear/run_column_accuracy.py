# pylint: skip-file
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
import argparse
import os

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.communication import init
from mindspore.ops.auto_generate import SumExt

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.tensor_parallel.layers import ColumnParallelLinear
from mindformers.experimental.utils import init_method_normal

inputs_ = ms.tensor(
    [[[0.5156250000, 0.3957519531, 0.0114746094, 0.5893554688],
      [0.2209472656, 0.5283203125, 0.4392089844, 0.3063964844],
      [0.1437988281, 0.7148437500, 0.3776855469, 0.8452148438],
      [0.9628906250, 0.5776367188, 0.3886718750, 0.5600585938],
      [0.9389648438, 0.1237182617, 0.7543945312, 0.0809326172],
      [0.8989257812, 0.7958984375, 0.9335937500, 0.4270019531],
      [0.7817382812, 0.0970458984, 0.0968017578, 0.1267089844],
      [0.8583984375, 0.7124023438, 0.3391113281, 0.8286132812]],

     [[0.1989746094, 0.7495117188, 0.7407226562, 0.6362304688],
      [0.1586914062, 0.9160156250, 0.7954101562, 0.3916015625],
      [0.9946289062, 0.1966552734, 0.7255859375, 0.7944335938],
      [0.3391113281, 0.0997924805, 0.3449707031, 0.1035766602],
      [0.1400146484, 0.4067382812, 0.3425292969, 0.3569335938],
      [0.7519531250, 0.9121093750, 0.5185546875, 0.4436035156],
      [0.6342773438, 0.4343261719, 0.5434570312, 0.7875976562],
      [0.3046875000, 0.6992187500, 0.5610351562, 0.5151367188]]],
    dtype=ms.float16)

weight = ms.tensor(
    [[-0.0245208740, -0.0001863241, 0.0308380127, -0.0093154907],
     [0.0055885315, -0.0052185059, 0.0124588013, -0.0231170654],
     [0.0023422241, -0.0377197266, 0.0436401367, -0.0038585663],
     [0.0107192993, -0.0177917480, -0.0061988831, 0.0154800415],
     [0.0024719238, -0.0436096191, 0.0074005127, 0.0082855225],
     [0.0371398926, 0.0395507812, -0.0086441040, 0.0273284912],
     [0.0168609619, -0.0008425713, 0.0331726074, -0.0261688232],
     [0.0199279785, 0.0187835693, 0.0282897949, 0.0126876831]],
    dtype=ms.float16)

bias = ms.tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=ms.float16)

output_pta = ms.tensor(
    [[[-0.0178527832, -0.0126647949, -0.0154953003, 0.0075378418,
       -0.0110168457, 0.0508117676, -0.0066833496, 0.0255126953],
      [0.0051727295, -0.0031337738, -0.0014257431, -0.0050125122,
       -0.0167083740, 0.0336914062, 0.0098342896, 0.0306396484],
      [0.0001142025, -0.0177612305, -0.0134048462, -0.0004341602,
       -0.0210266113, 0.0534362793, -0.0077667236, 0.0376892090],
      [-0.0169525146, -0.0057373047, -0.0047340393, 0.0063056946,
       -0.0152969360, 0.0705566406, 0.0139846802, 0.0481262207],
      [-0.0005373955, 0.0121307373, 0.0301361084, 0.0044403076,
       0.0031795502, 0.0354614258, 0.0386352539, 0.0433959961],
      [0.0026226044, 0.0026302338, 0.0111770630, -0.0037021637,
       -0.0220336914, 0.0684814453, 0.0342712402, 0.0646972656],
      [-0.0173797607, 0.0021400452, 0.0019063950, 0.0080108643,
       -0.0005335808, 0.0354919434, 0.0129928589, 0.0217437744],
      [-0.0184478760, -0.0138473511, -0.0132598877, 0.0072517395,
       -0.0195770264, 0.0797729492, 0.0034389496, 0.0505981445]],

     [[0.0118942261, -0.0082778931, 0.0020656586, -0.0059432983,
       -0.0214385986, 0.0480041504, 0.0106430054, 0.0470581055],
      [0.0168151855, -0.0030364990, -0.0009794235, -0.0134658813,
       -0.0304260254, 0.0459594727, 0.0180358887, 0.0478515625],
      [-0.0094528198, -0.0047912598, 0.0235137939, 0.0149612427,
       0.0058364868, 0.0601501465, 0.0198822021, 0.0541076660],
      [0.0013399124, 0.0032787323, 0.0116882324, 0.0013246536,
       -0.0001025200, 0.0163879395, 0.0143661499, 0.0196990967],
      [0.0037288666, -0.0053253174, -0.0014429092, -0.0023345947,
       -0.0119018555, 0.0280761719, 0.0040397644, 0.0246429443],
      [-0.0067481995, -0.0043525696, -0.0117263794, -0.0045166016,
       -0.0304107666, 0.0716552734, 0.0175018311, 0.0524291992],
      [-0.0062103271, -0.0101547241, 0.0057792664, 0.0078964233,
       -0.0068244934, 0.0575561523, 0.0077476501, 0.0461730957],
      [0.0049018860, -0.0068664551, -0.0031642914, -0.0046768188,
       -0.0213165283, 0.0481872559, 0.0096817017, 0.0416259766]]],
    dtype=ms.float16)

inputs_grad = ms.tensor(
    [[[0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926]],

     [[0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926],
      [0.0705566406, -0.0470275879, 0.1409912109, 0.0013217926]]],
    dtype=ms.float16)

weight_grad = ms.tensor(
    [[8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500],
     [8.8437500000, 8.3593750000, 7.9140625000, 7.7929687500]],
    dtype=ms.float16)

bias_grad = ms.tensor([16., 16., 16., 16., 16., 16., 16., 16.], dtype=ms.float16)


class MyNet(nn.Cell):
    def __init__(self, input_size: int, output_size: int, config: TransformerConfig):
        super(MyNet, self).__init__()
        net = ColumnParallelLinear(input_size=input_size, output_size=output_size,
                                   config=config, compute_dtype=ms.dtype.float16,
                                   init_method=init_method_normal(), transpose_b=True, bias=True)
        net.weight.set_data(weight)
        net.bias.set_data(bias)
        self.linear = net

    def construct(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        output, _ = self.linear(inputs)
        return output


class TestNet(nn.Cell):
    def __init__(self, input_size: int, output_size: int, config: TransformerConfig):
        super(TestNet, self).__init__()
        net = ColumnParallelLinear(input_size=input_size, output_size=output_size,
                                   config=config, compute_dtype=ms.dtype.float16,
                                   init_method=init_method_normal(), transpose_b=True, bias=True)
        net.weight.set_data(weight)
        net.bias.set_data(bias)
        self.linear = net
        self.sum = SumExt()

    def construct(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        output, _ = self.linear(inputs)
        all_sum = self.sum(output)
        return all_sum


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dp',
        default=1,
        type=int,
        required=False,
        help='data_parallel')
    parser.add_argument(
        '--cp',
        default=1,
        type=int,
        required=False,
        help='context_parallel')
    parser.add_argument(
        '--tp',
        default=1,
        type=int,
        required=False,
        help='tensor_parallel')
    parser.add_argument(
        '--skip_weight',
        action='store_true',
        help='skip_weight_param_allocation'
    )
    parser.add_argument(
        '--has_bias',
        action='store_true',
        help='has_bias'
    )
    args, _ = parser.parse_known_args()
    return args


def get_config(args):
    config = TransformerConfig()
    config.data_parallel = args.dp
    config.tensor_parallel = args.tp
    config.context_parallel = args.cp
    return config


def do_init():
    ms.set_context(deterministic="ON", mode=ms.GRAPH_MODE)
    rank_id = os.environ.get('RANK_ID')
    if rank_id is not None:
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
        init()
    return rank_id


def main():
    rank_id = do_init()
    input_size = inputs_.shape[-1]
    output_size = output_pta.shape[-1]
    args = get_args()
    config = get_config(args)

    my_net = MyNet(input_size, output_size, config)
    out_res = my_net(inputs_)
    assert output_pta.equal(out_res).all()

    test_net = TestNet(input_size, output_size, config)
    test_net = test_net.to_float(ms.float16)
    grad_func = ms.grad(test_net, grad_position=0, weights=test_net.trainable_params(), has_aux=False)
    grad_res = grad_func(inputs_)
    tols = {'atol': 1e-3, 'rtol': 1e-3}

    inputs_grad_ms = grad_res[0]
    if rank_id is not None:
        reduce = ms.ops.AllReduce()
        inputs_grad_ms = reduce(grad_res[0])
    assert ms.ops.isclose(inputs_grad, inputs_grad_ms, rtol=tols['rtol'], atol=tols['atol']).all()

    weight_grad_ms = grad_res[1][0]
    if rank_id is not None:
        gather = ms.ops.AllGather()
        weight_grad_ms = gather(grad_res[1][0])
    assert ms.ops.isclose(weight_grad_ms, weight_grad, rtol=tols['rtol'], atol=tols['atol']).all()

    bias_grad_ms = grad_res[1][1]
    if rank_id is not None and args.tp > 1:
        gather = ms.ops.AllGather()
        bias_grad_ms = gather(grad_res[1][1])
    assert bias_grad_ms.equal(bias_grad).all()


if __name__ == "__main__":
    main()
