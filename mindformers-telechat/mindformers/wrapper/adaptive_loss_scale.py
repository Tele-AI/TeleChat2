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
"""Loss scale cell for loss scale training."""
from __future__ import absolute_import

import numpy as np
import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.common import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['AdaptiveLossScaleUpdateCell']


def _get_window_list(max_scale_window, min_scale_window, window_interval, window_factor):
    """ automatic generate the scale window list with max_scale_window and min_scale_window. """
    window_list = []
    window_list.append(int(max_scale_window))
    while max_scale_window > min_scale_window:
        if max_scale_window > window_interval:
            max_scale_window = int(max_scale_window / window_factor / window_interval) * window_interval
        elif max_scale_window == window_interval:
            max_scale_window = max_scale_window / window_factor
        else:
            max_scale_window = max_scale_window - min_scale_window / window_factor
        window_list.append(int(max_scale_window))
    window_list.reverse()
    window_list_arr = np.array(window_list)
    return Tensor(window_list_arr, dtype=mstype.int32), window_list


def _get_list_index(window_list, scale_window):
    """ get the init scale window list index with input scale_window. """
    # if scale window is not int the list, set the index to 0
    window_list = window_list.asnumpy().tolist()
    if scale_window in window_list:
        list_index = window_list.index(scale_window)
    else:
        logger.warning("scale_window is not in the generated window list, "
                       "will use min_scale_window to start the training.")
        list_index = 0
    return list_index


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class AdaptiveLossScaleUpdateCell(Cell):
    r"""
    Adaptive Loss scale update cell.

    For loss scaling training, the initial loss scaling value will be set to be `loss_scale_value`.
    A scale window list which will be used to control loss scale adaptively will be initialized
    according to 'max_scale_window'.
    In each training step, the loss scaling value will be decreased by `loss_scale`/`scale_factor`
    when there is an overflow. And it will be increased by `loss_scale` * `scale_factor` if there is no
    overflow for a continuous `scale_window` steps. Moreover, the scale window will be increased to next
    level if loss_scale increases three times during current scale window. The scale
    window will be decreased to '1' if loss_scale decreases three times consecutively.

    Args:
        loss_scale_value (float): Initializes loss scale.
        scale_factor (int): Coefficient of increase and decrease.
        scale_window (int): current Maximum continuous training steps that do not have overflow to increase loss scale.
        max_scale_window (int): Maximum scale_window of the automatic scale window list. The default value is 20.
        min_scale_window (int): Minimum scale_window of the automatic scale window list. The default value is 1000.

    Inputs:
        - **loss_scale** (Tensor) - The loss scale value during training with shape :math:`()`.
        - **overflow** (bool) - Whether the overflow occurs or not.

    Outputs:
        bool, the input `overflow`.

    Supported Platforms:
        ``Ascend``
    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset import GeneratorDataset
        >>> from mindspore.nn import Momentum
        >>> from mindformers import Trainer, TrainingArguments, AutoModel
        >>> from mindformers import init_context, ContextConfig
        >>> from mindformers.wrapper import MFTrainOneStepCell, AdaptiveLossScaleUpdateCell
        >>>
        >>>
        >>> def context_init():
        >>>     context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
        >>>     rank_id, device_num = init_context(use_parallel=False, context_config=context_config)
        >>>
        >>>
        >>> def generator():
        >>>     seq_len = 1025
        >>>     input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
        >>>     for _ in range(512):
        >>>         yield input_ids
        >>>
        >>> # 环境初始化
        >>> context_init()
        >>> # 自定义训练超参数
        >>> training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001,
        >>>                                 warmup_steps=1000, sink_mode=True)
        >>> # 自定义模型
        >>> pangu_model = AutoModel.from_pretrained("pangualpha_2_6b")
        >>> opt = Momentum(learning_rate=0.1, momentum=0.9,
        >>>             params=pangu_model.trainable_params(),)
        >>> manager = AdaptiveLossScaleUpdateCell(loss_scale_value=212, scale_factor=2, scale_window=20,
        >>>                                       max_scale_window=1000, min_scale_window=20)
        >>> train_network = MFTrainOneStepCell(pangu_model, opt, scale_sense=manager)
        >>> train_network.set_train()
        >>> # 自定义数据集
        >>> dataset = GeneratorDataset(generator, column_names=["input_ids"])
        >>> train_dataset = dataset.batch(batch_size=4)
        >>> eval_dataset = dataset.batch(batch_size=4)
        >>> # 定义文本生成任务，传入自定义模型、数据集、超参数
        >>> text_generation = Trainer(task='text_generation', model_name='pangualpha_2_6b',
        >>>                         wrapper=train_network, args=training_args,
        >>>                         train_dataset=train_dataset, eval_dataset=eval_dataset)
    """

    def __init__(self,
                 loss_scale_value,
                 scale_factor,
                 scale_window,
                 max_scale_window=1000,
                 min_scale_window=20
                 ):
        super(AdaptiveLossScaleUpdateCell, self).__init__()

        if max_scale_window <= 0 or min_scale_window <= 0 or max_scale_window <= min_scale_window:
            raise ValueError(f"`max_scale_window` and `min_scale_window` have to be floats > 0 and `max_scale_window` "
                             f"has to be larger than `min_scale_window`")

        if not isinstance(max_scale_window, int):
            raise TypeError(f"max_scale_window should be a int, but got {type(max_scale_window)}")

        if not isinstance(min_scale_window, int):
            raise TypeError(f"min_scale_window should be a int, but got {type(min_scale_window)}")
        self.max_scale_window = max_scale_window
        self.min_scale_window = min_scale_window
        self.window_interval = 100
        self.window_factor = 2
        self.const_update_threshold = Tensor(3, dtype=mstype.int32)
        self.const_mod_interval = Tensor(4, dtype=mstype.int32)
        self.const_add_interval = Tensor(1, dtype=mstype.int32)
        self.const_init_value = Tensor(0, dtype=mstype.int32)
        self.window_list, self.window_list_num = _get_window_list(self.max_scale_window, self.min_scale_window,
                                                                  self.window_interval, self.window_factor)
        self.window_list_len = Tensor(len(self.window_list) - 1, dtype=mstype.int32)
        self.list_index_num = _get_list_index(self.window_list, scale_window)
        self.scale_window = Parameter(Tensor(self.window_list_num[int(self.list_index_num)], dtype=mstype.int32),
                                      name="scale_window")
        self.invalid_window_list_index = Tensor(-1, dtype=mstype.int32)
        self.list_index = Parameter(Tensor(int(self.list_index_num), dtype=mstype.int32), name="list_index")
        self.window_up_count = Parameter(Tensor(0, dtype=mstype.int32), name="scale_window_up_count")
        self.window_down_count = Parameter(Tensor(0, dtype=mstype.int32), name="scale_window_down_count")
        self.scale_factor = Tensor(scale_factor, dtype=mstype.float32)
        self.loss_scale_value = loss_scale_value
        self.cur_iter = Parameter(Tensor(1, dtype=mstype.int32), name="current_iterator_step")
        self.last_overflow_iter = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        self.select = P.Select()
        self.max = P.Maximum()
        self.minimum_loss_scale = Tensor(1.0, dtype=mstype.float32)
        self.reciprocal = P.Reciprocal()
        self.equal = P.Equal()
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.logic_and = P.LogicalAnd()
        self.logic_not = P.LogicalNot()
        self.logic_or = P.LogicalOr()
        self.const_true = Tensor(True, dtype=mstype.bool_)
        self.mod = P.Mod()
        self.add = P.Add()
        self.cast = P.Cast()

    def get_loss_scale(self):
        """
        Get Loss Scale value.

        Returns:
            float, the loss scale value.

        Examples:
            >>> from mindformers.wrapper import AdaptiveLossScaleUpdateCell
            >>> manager = AdaptiveLossScaleUpdateCell(loss_scale_value=212, scale_factor=2, scale_window=1000,
            >>>                                       max_scale_window=1000, min_scale_window=20)
            >>> output = manager.get_loss_scale()
            >>> print(output)
            212
        """
        return self.loss_scale_value

    def construct(self, loss_scale, overflow):
        """
        Inputs:
            - **loss_scale** (Tensor) - The loss scale value during training with shape :math:`()`.
            - **overflow** (bool) - Whether the overflow occurs or not.

        Outputs:
            bool, the input `overflow`.
        """
        overflow_cond = overflow
        loss_scale_on_overflow = self.select(overflow_cond, self.max(loss_scale * self.reciprocal(self.scale_factor),
                                                                     self.minimum_loss_scale), loss_scale)
        should_inc = self.less_equal(self.scale_window, self.cur_iter - self.last_overflow_iter)
        last_iter_cond = self.logic_or(overflow_cond, should_inc)
        last_overflow_iter = self.select(last_iter_cond, self.cur_iter, self.last_overflow_iter)
        last_iter = F.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        F.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        inc_cur_iter = F.depend(inc_cur_iter, last_iter)
        F.assign(self.cur_iter, inc_cur_iter)
        # if self.window_up_count equals to self.const_update_threshold, use the next level scale window
        up_num_update = self.mod(self.add(self.window_up_count, self.const_add_interval), self.const_mod_interval)
        F.assign(self.window_up_count, self.select(update_scale_cond, up_num_update, self.window_up_count))
        F.assign(self.window_down_count, self.select(update_scale_cond, self.const_init_value, self.window_down_count))
        window_up = self.equal(self.window_up_count, self.const_update_threshold)
        list_index_cond = self.not_equal(self.list_index, self.window_list_len)
        window_up_cond = self.logic_and(window_up, list_index_cond)
        list_index_update = self.add(self.list_index, self.const_add_interval)
        F.assign(self.list_index, self.select(window_up_cond, list_index_update, self.list_index))
        F.assign(self.window_up_count, self.select(window_up_cond, self.const_init_value, self.window_up_count))
        cur_list_index = self.select(self.equal(self.list_index, self.invalid_window_list_index), self.const_init_value,
                                     self.list_index)
        F.assign(self.scale_window, self.select(window_up_cond, self.window_list[self.cast(cur_list_index, ms.int32)],
                                                self.scale_window))
        # if self.window_down_count equals to self.const_update_threshold, set scale window to 1
        down_num_update = self.mod(self.add(self.window_down_count, self.const_add_interval), self.const_mod_interval)
        F.assign(self.window_down_count, self.select(overflow_cond, down_num_update, self.window_down_count))
        window_down = self.logic_and(overflow_cond, self.equal(self.window_down_count, self.const_update_threshold))
        down_check = self.logic_and(self.not_equal(self.list_index, self.invalid_window_list_index),
                                    self.not_equal(self.list_index, self.const_init_value))
        window_down_cond = self.logic_and(window_down, down_check)
        F.assign(self.scale_window, self.select(window_down_cond, self.const_add_interval, self.scale_window))
        F.assign(self.list_index, self.select(window_down_cond, self.invalid_window_list_index, self.list_index))
        F.assign(self.window_down_count, self.select(window_down_cond, self.const_init_value, self.window_down_count))
        F.assign(self.window_up_count, self.select(window_down_cond, self.const_init_value, self.window_up_count))
        return overflow
