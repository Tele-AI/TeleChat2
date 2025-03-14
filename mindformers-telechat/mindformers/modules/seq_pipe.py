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
"""seq pipe wrapper."""
from mindspore import nn
from mindspore.ops import operations as P
from mindformers.wrapper.wrapper import DataOrderWrapperCell


class _MicroBatchByAxis(nn.Cell):
    """MicroBatchByAxis"""
    def __init__(self, micro_size, seq_dim=1):
        super(_MicroBatchByAxis, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.micro_size = micro_size
        self.strided_slice = P.StridedSlice()
        self.seq_dim = seq_dim
        self.depend = P.Depend()

    def construct(self, i, return_depend, *inputs):
        """_MicroBatchByAxis construct"""
        micro_inputs = ()
        for each_input in inputs:
            each_input = self.depend(each_input, return_depend)
            input_shape = self.shape(each_input)
            seq_input_shape = input_shape
            if seq_input_shape[self.seq_dim] % 2 != 0:
                seq_len = seq_input_shape[self.seq_dim] - 1
                micro_batch_begin = i * seq_len // self.micro_size
                micro_batch_end = (i + 1) * seq_len // self.micro_size + 1
            else:
                seq_len = seq_input_shape[self.seq_dim]
                micro_batch_begin = i * seq_len // self.micro_size
                micro_batch_end = (i + 1) * seq_len // self.micro_size
            strided_slice_begin = ()
            strided_slice_strides = ()
            strided_slice_end = ()
            for j, _ in enumerate(seq_input_shape):
                strided_slice_strides += (1,)
                if j == self.seq_dim:
                    strided_slice_begin += (micro_batch_begin,)
                    strided_slice_end += (micro_batch_end,)
                else:
                    strided_slice_begin += (0,)
                    strided_slice_end += (seq_input_shape[j],)
            micro_input = self.strided_slice(each_input, strided_slice_begin, strided_slice_end, strided_slice_strides)
            micro_inputs += (micro_input,)

        return micro_inputs


class SequenceSplit(nn.Cell):
    """
    This function splits the input at the sequence dimension into split_num pieces
    then performs the computation of the wrapped cell.
    """

    def __init__(self, network, split_num=2, seq_dim=1):
        super(SequenceSplit, self).__init__(auto_prefix=False)
        if not isinstance(split_num, int):
            raise TypeError("For 'SequenceSplit', the argument 'split_num' must be integer, "
                            "but got {}.".format(split_num))
        if split_num <= 0:
            raise ValueError("For 'SequenceSplit', the argument 'split_num' must be larger than 0, "
                             "but got {}.".format(split_num))
        self.network = network
        self.split_num = split_num
        self.interleave_inputs = nn.CellList()
        # Add attr for mindspore finding the add.
        self.add = P.Add().add_prim_attr("seq_split_add", True)
        self.depend = P.Depend()
        self.with_data_order_wrapper = isinstance(network, DataOrderWrapperCell)
        for _ in range(split_num):
            interleave_data = _MicroBatchByAxis(split_num, seq_dim)
            # Add attr for mindspore finding the slice and get the split num.
            interleave_data.strided_slice.add_prim_attr("strided_slice_flag", True)
            interleave_data.strided_slice.add_prim_attr("seq_split_slice", True)
            interleave_data.strided_slice.add_prim_attr("split_num", split_num)
            self.interleave_inputs.append(interleave_data)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """ seqpipe construct"""
        output = 0.0
        div_num = 1e-9
        output1 = 0.0
        div_num1 = 1e-9
        extra_loss = 0
        if not self.with_data_order_wrapper:
            return_depend = self.network.clear_kv_cache()
        else:
            return_depend = self.network.network.clear_kv_cache()
        for i in range(self.split_num):
            interleave_input = self.interleave_inputs[i](i, return_depend, *inputs)
            outs = self.network(*interleave_input)
            if len(outs) == 2:
                numerator, denominator = outs
                output = self.add(output, numerator)
                div_num = self.add(div_num, denominator)
            elif len(outs) == 5:
                numerator, denominator, numerator1, denominator1, ex_loss = outs
                output = self.add(output, numerator)
                div_num = self.add(div_num, denominator)
                output1 = self.add(output1, numerator1)
                div_num1 = self.add(div_num1, denominator1)
                extra_loss = self.add(extra_loss, ex_loss)
        return output / div_num + output1 / div_num1 + extra_loss
