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
"""generate MoE token distribution charts script"""
import argparse
import os
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt


def append_all_npy_path(path):
    """
    Append all npy files path to path_list

    Args:
    path(string): The path where the npy files are stored.

    Returns:
    path_list(list): A list of npy files path
    """
    path_list_temp = []
    for root, _, files in os.walk(path):
        for each in files:
            real_path = (os.path.join(root, each))
            path_list_temp.append(real_path)
    return path_list_temp


def merge_npy_files(str_layer, path_npy_list):
    """
    Merge the npy files of each MOE layer into a complete file

    Args:
    str_layer(string): Layer number (for example layer-0、layer-1)
    path_npy_list(list): A list of npy files path

    Returns:
    data(list): Token distribution data of a moe layer
    """
    sub_path_list = []
    for npy_list in path_npy_list:
        if str_layer in npy_list:
            sub_path_list.append(npy_list)
    if not sub_path_list:
        raise Exception(f"error: {str_layer} is not exist")
    sub_path_list.sort(key=lambda x: int(x.split(str_layer)[1].split('.')[0]))
    temp = []
    for path in sub_path_list:
        real_data = np.load(path, allow_pickle=True)
        temp.append(real_data)
    return temp


def pyplot_show(str_layer, layer_capital, layer_data, save_path_prefix):
    """
    Generate the picture shows the distribution of hot and cold experts

    Args:
    str_layer(string): Layer number (for example layer-0、layer-1)
    layer_capital(string): Capitalize first letter of str_layer (for example Layer-0、Layer-1)
    data(list): Token distribution data of a MoE layer
    save_path_prefix(string): The save path prefix for saving the picture

    Returns:
    """
    if not data:
        raise Exception(f"{str_layer} data is empty")
    layer_data = ms.Tensor(layer_data, dtype=ms.float16)
    layer_data = ms.ops.transpose(layer_data, (1, 0))
    expert_num = layer_data.shape[0]
    step_num = layer_data.shape[1]
    # Horizontal coordinate of the point
    x = np.arange(step_num).reshape(-1,)
    layer_data_list = []
    str_expert_list = []
    for i in range(expert_num):
        # Vertical coordinate of the point
        ki = layer_data[i].asnumpy()
        layer_data_list.append(ki)
        str_expert = 'expert-' + str(i)
        str_expert_list.append(str_expert)
    layer_data_tuple = tuple(layer_data_list)
    fig = plt.figure(figsize=(10, 4), dpi=500)
    ax = fig.add_subplot(1, 1, 1)
    plt.stackplot(x, layer_data_tuple, labels=str_expert_list)
    # Horizontal coordinate name
    plt.xlabel("num of step")
    # Vertical coordinate name
    plt.ylabel("num of token")
    # The title
    plt.title(layer_capital + '_Token_Distribution', fontsize='large', fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    # Legend
    ax.legend(handles[::-1], labels[::-1], fontsize='small', loc=7, borderaxespad=-7)
    plt.show()
    save_path = save_path_prefix + str_layer + '_token_distribution.jpg'
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and merge npy files')

    parser.add_argument(
        '--num_layers', type=int, default=6,
        help='The number of layers of the model to be converted.')

    parser.add_argument(
        '--hot_expert_num', type=int, default=2,
        help='The number of hot expert for one MoE layer')

    parser.add_argument(
        '--npy_files_load_path', type=str, default='../summary_dir/summary_baseline/export/tensor',
        help='The path where the npy files are stored')

    parser.add_argument(
        '--save_path_prefix', type=str, default='./token_distribution/',
        help='The path prefix after merging npy files')

    opt = parser.parse_args()
    if not os.path.exists(opt.save_path_prefix):
        os.makedirs(opt.save_path_prefix)
    path_list = append_all_npy_path(opt.npy_files_load_path)
    list_all = []
    for layer_num in range(opt.num_layers):
        str_layer_split = 'layer-' + str(layer_num) + '_'
        str_layer_name = 'layer-' + str(layer_num)
        str_layer_capital = 'Layer-' + str(layer_num)
        data = merge_npy_files(str_layer_split, path_list)
        last_step_data = ms.Tensor(data[-1], dtype=ms.float16)
        _, hot_expert_index = ms.ops.TopK(sorted=True)(last_step_data, opt.hot_expert_num)
        pyplot_show(str_layer_name, str_layer_capital, data, opt.save_path_prefix)
        print(f"{str_layer_name} token_distribution generate successful")
        hot_expert_index_list = []
        for j in range(opt.hot_expert_num):
            hot_expert_index_list.append(int(str(hot_expert_index[j])))
        list_all.append(hot_expert_index_list)
        if layer_num == opt.num_layers - 1:
            print("All layers token_distribution generate successful")
    print("hot_expert_index: ", list_all)
