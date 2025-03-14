# pylint: skip-file
import argparse
import os
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, dtype
from mindspore.communication import init

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.rotary_pos_embedding import ApplyRotaryPosEmb


class MyNet(nn.Cell):
    def __init__(self, config: TransformerConfig):
        super(MyNet, self).__init__()
        self.emb = ApplyRotaryPosEmb(config)

    def construct(self, t: Tensor, freqs: Tensor, rotary_interleaved: bool = False):
        output = self.emb(t, freqs, rotary_interleaved)
        return output


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

    seed_value = 42
    ms.set_seed(seed_value)
    np.random.seed(seed_value)


def main():
    args = get_args()
    config = get_config(args)

    bs, n_heads, seq_len, head_dim = 2, 8, 512, 8
    input_shape = (bs, n_heads, seq_len, head_dim)
    freqs_shape = (1, 1, seq_len, head_dim)

    net = MyNet(config)
    input_ = ms.tensor(np.random.standard_normal(input_shape), dtype.float32)
    freqs_ = ms.tensor(np.random.standard_normal(freqs_shape), dtype.float32)
    output = net(input_, freqs_)

    print(output.shape)


if __name__ == "__main__":
    main()
