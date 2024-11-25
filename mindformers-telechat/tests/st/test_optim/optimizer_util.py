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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
# ============================================================================
"""Optimizer util."""
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.ops import operations as P

from mindformers.core.optim import build_optim

np.random.seed(1024)

fc1_weight = np.array([[0.72346634, 0.95608497, 0.4084163, 0.18627149,
                        0.6942514, 0.39767185, 0.24918061, 0.4548748],
                       [0.7203382, 0.19086994, 0.76286614, 0.87920564,
                        0.3169892, 0.9462494, 0.62827677, 0.27504718],
                       [0.3544535, 0.2524781, 0.5370583, 0.8313121,
                        0.6670143, 0.0488653, 0.62225235, 0.7546456],
                       [0.17985944, 0.05106374, 0.31064633, 0.4863033,
                        0.848814, 0.5523157, 0.20295663, 0.7213356]]).astype("float32")

fc1_bias = np.array([0.79708564, 0.13728078, 0.66322654, 0.88128525]).astype("float32")

fc2_weight = np.array([[0.8473515, 0.50923985, 0.42287776, 0.29769543]]).astype("float32")

fc2_bias = np.array([0.09996348]).astype("float32")


def make_fake_data():
    """
    make fake data
    """
    data, label = [], []
    for i in range(20):
        data.append(mindspore.Tensor(np.array(np.ones((2, 8)) * i, dtype=np.float32)))
        label.append(mindspore.Tensor(np.array(np.ones((2, 1)) * (i + 1), dtype=np.float32)))
    return data, label


class NetWithLoss(nn.Cell):
    """
    build net with loss
    """

    def __init__(self, network, loss_fn):
        super(NetWithLoss, self).__init__()
        self.network = network
        self.loss = loss_fn

    def construct(self, x, label):
        out = self.network(x)
        loss = self.loss(out, label)
        return loss


class FakeNet(nn.Cell):
    """
    build fake net
    """

    def __init__(self):
        super(FakeNet, self).__init__()
        self.fc1 = nn.Dense(in_channels=8, out_channels=4, weight_init=Tensor(fc1_weight), bias_init=Tensor(fc1_bias))
        self.fc2 = nn.Dense(in_channels=4, out_channels=1, weight_init=Tensor(fc2_weight), bias_init=Tensor(fc2_bias))
        self.relu = nn.ReLU()
        self.reducemean = P.ReduceMean()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        """
        parameter initialization
        """
        self.init_parameters_data()
        for name, m in self.cells_and_names():
            if name == 'fc1':
                m.weight.set_data(Tensor(fc1_weight))
                m.bias.set_data(Tensor(fc1_bias))
            elif name == 'fc2':
                m.weight.set_data(Tensor(fc2_weight))
                m.bias.set_data(Tensor(fc2_bias))


def build_network(opt_config, net, is_group=None, loss_fn=None):
    """
    Construct training
    """
    if is_group is None:
        is_group = False
    if loss_fn is None:
        loss_fn = nn.L1Loss(reduction='mean')
    losses = []
    networkwithloss = NetWithLoss(net, loss_fn)
    networkwithloss.set_train()

    if is_group:
        fc1_params = list(filter(lambda x: 'fc1' in x.name, networkwithloss.trainable_params()))
        fc2_params = list(filter(lambda x: 'fc1' not in x.name, networkwithloss.trainable_params()))
        if opt_config['type'] == 'AdamW':
            params = [{'params': fc1_params, 'weight_decay': 0.01, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.1}]
        else:
            params = [{'params': fc1_params, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.01}]
    else:
        params = networkwithloss.trainable_params()

    opt_config['params'] = params

    net_opt = build_optim(opt_config)
    trainonestepcell = mindspore.nn.TrainOneStepCell(networkwithloss, net_opt)
    data, label = make_fake_data()
    for i in range(20):
        loss = trainonestepcell(data[i], label[i])
        losses.append(loss.asnumpy())
    return np.array(losses), net_opt


default_fc1_weight_adamw_m = (
    np.array([[0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935],
              [0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492],
              [0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864],
              [-0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057,
               -0.06905057]], dtype=np.float32)
)

default_fc2_weight_adamw_m = (
    np.array([[6.978479, 7.470356, 5.508465, 5.176325]], dtype=np.float32)
)

default_fc1_weight_adamw_v = (
    np.array([[0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346],
              [0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226],
              [0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351],
              [0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008]],
             dtype=np.float32)
)

default_fc2_weight_adamw_v = (
    np.array([[35.217834, 42.283375, 26.52298, 21.510029]], dtype=np.float32)
)
