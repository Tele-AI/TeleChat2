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
"""glm32k predict example."""
import os
import argparse

import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, logger
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration, ChatGLM3Tokenizer
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint


def main(config_path, use_parallel, load_checkpoint, vocab_file):
    inputs = ["晚上睡不着应该怎么办", "使用python编写快速排序代码"]
    batch_size = len(inputs)

    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = use_parallel
    device_num = os.getenv('MS_WORKER_NUM')
    logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint

    # init context
    build_context(config)
    build_parallel_config(config)

    # init model
    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = ChatGLM2Config(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # init tokenizer
    tokenizer = ChatGLM3Tokenizer(vocab_file=vocab_file)

    # build model
    network = ChatGLM2ForConditionalGeneration(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        # set auto transform ckpt
        if os.path.isdir(config.load_checkpoint) or config.use_parallel:
            config.auto_trans_ckpt = True
        else:
            config.auto_trans_ckpt = False
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict using generate
    if isinstance(inputs, list):
        inputs_ids = tokenizer.build_batch_input(inputs)["input_ids"]
    else:
        inputs_ids = tokenizer.build_chat_input(inputs)["input_ids"]
    outputs = network.generate(inputs_ids,
                               max_length=model_config.max_decode_length,
                               do_sample=model_config.do_sample,
                               top_k=model_config.top_k,
                               top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_glm32k.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--vocab_file', type=str,
                        help='tokenizer.model file path.')
    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.vocab_file
    )

# [gMASK]sop<|user|>
# 晚上睡不着应该怎么办<|assistant|>
# 晚上睡不着,可以参考下述建议:
# 1. 建立规律的睡眠时间表:每天在相同的时间上床和起床,有助于身体建立规律的睡眠时间表,更容易入睡。
# 2. 创造舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗,凉爽,有助于入睡。
# 3. 避免刺激性物质:避免饮用咖啡因和酒精等刺激性物质,这些物质会影响睡眠。
# 4. 放松身心:在睡前放松身心,例如泡个热水澡,听些轻柔的音乐,读本书等,有助于入睡。
# 5. 避免使用电子设备:在睡前避免使用电子设备,例如手机,平板电脑等,这些设备发出的蓝光会抑制睡眠激素的分泌,影响睡眠。
# 6. 锻炼身体:适度的锻炼身体有助于睡眠,但避免在睡前进行剧烈运动。
# 7. 寻求专业帮助:如果长期存在睡眠问题,建议寻求专业医生的帮助。
#
# 如果以上建议都无法解决问题,建议咨询医生,了解更具体的解决方案。
# [gMASK]sop<|user|>
# 使用python编写快速排序代码<|assistant|>
# 快速排序（Quick Sort）是一种高效的排序算法，其基本思想是通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分关键字小，然后分别对这两部分记录继续进行排序，以达到整个序列有序。
#
# 下面是使用 Python 编写的快速排序代码：
#
# ```python
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quick_sort(left) + middle + quick_sort(right)
#
# arr = [3,6,8,10,1,2,1]
# print("原始数组：", arr)
# print("排序后的数组：", quick_sort(arr))
# ```
#
# 这段代码首先定义了一个名为 `quick_sort` 的函数，该函数接受一个列表作为参数。然后，我们选择列表中间的元素作为基准值（pivot），
# 并将列表中的元素分为三部分：小于基准值的元素（left）、等于基准值的元素（middle）和大于基准值的元素（right）。
# 最后，我们递归地对左右两部分进行快速排序，并将排序后的结果合并在一起。
#
# 运行这段代码，输出结果如下：
#
# ```
# 原始数组： [3, 6, 8, 10, 1, 2, 1]
# 排序后的数组： [1, 1, 2, 3, 6, 8, 10]
# ```
#
# 这就是使用 Python 编写的快速排序代码。
