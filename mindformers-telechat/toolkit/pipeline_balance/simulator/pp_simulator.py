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
r"""pipeline simulator"""
from __future__ import annotations
import copy
import sys
import numpy as np

from toolkit.pipeline_balance.simulator.sim_block import BlockSim, SendBlockSim, RecBlockSim
from toolkit.pipeline_balance.simulator.pipeline_builder import PipelineBuilder
from toolkit.pipeline_balance.simulator.causal_error import CausalCommError, CausalError
from toolkit.pipeline_balance.simulator.plot_manager import PlotMgr
from toolkit.pipeline_balance.simulator.utils import format_2d_inputs, apply_format, apply_color

sys.setrecursionlimit(8192)


class PipelineSimulator:
    r"""
    Pipeline Simulator which provide pipeline flow process, bubbles and relative memories for stages.

    Args:
        block_time(Union[List[int|float], List[List[int|float]]]): relative forward computing time for each block.
            If it is List of List, the outer List indicates number of virtual-pp
            while the inner List indicates pp_stage.
        micro_num(int): micro batch number.
        comm_time(float): communication block (send/receive) time. Default: 0.1
        layer_recompute(Union[bool, List[int|float], List[List[int|float]]]): the block recompute information.
            If it is bool type, the backward block will be extended by block_time depending on whether it is True.
            Otherwise it represents relative computing time of recompute for each block. Default: False.
        block_mem(Union[bool, List[int|float], List[List[int|float]]]): the block memory information.
            If it is a number, the memory will be `block_mem` * `block_time`. Otherwise it represents relative memory
            for each block. Default: 1.
        backward_ratio(Union[List[int|float], List[List[int|float]]]): the ratios of backward computing time
        and forward computing time for each block. Default: 2.

    Example:
        A PipelineSimulator with pp=4, micro=16, each stage has 8 layers and last stage has extra head and
        loss computation equivalent to 0.8 layer:
        >>> sim = PipelineSimulator([8,8,8,8+0.8], 16, comm_time=0.1)       # create an instance of PipelineSimulator
        >>> sim.run()       # run simulation to scheduler the pipeline (information will be automatically printed)
        ————————————— pp: 4, vp: 1, micro: 16 ————————————
        --------------------  bubble  --------------------
        real    =   ideal   +   imba    +   comm
        0.2658   =  0.1875   +  0.0615   +  0.0168
        --------------------  memory  --------------------
        peak memory: 32.00, 24.00, 16.00, 8.80
        >>> sim.show()      # draw the pipeline and memory timeline picture

        Show imbalance timeline of vp=2, pp=4, micro=8, total 16 layers with extra equivalent 1.2 layer:
        >>> PipelineSimulator([[2,2,2,2],[1,2,3,2+1.2]], 8, comm_time=0.1).run().show()
        ————————————— pp: 4, vp: 2, micro:  8 ————————————
        --------------------  bubble  --------------------
        real    =   ideal   +   imba    +   comm
        0.4971   =  0.1875   +  0.2447   +  0.0649
        --------------------  memory  --------------------
        peak memory: 18.00, 18.00, 18.00, 14.80

        Show timeline of vp=3, pp=8, micro=16, total 48 layers with extra equivalent 0.6 layer.
        some of layers are recomputed and set memory correspondingly:
        >>> PipelineSimulator([[2,2,2,2,2,2,2,2],
        >>>                    [2,2,2,2,2,2,2,2],
        >>>                    [2,2,2,2,2,2,2,2+0.6]], 16, 0.1,
        >>>                    [[0,0,0,0,0,0,0,0],
        >>>                    [1,0,0,0,0,0,0,0],
        >>>                    [2,2,1,0,0,0,0,0]],
        >>>                    [[2,2,2,2,2,2,2,2],
        >>>                    [1.1,2,2,2,2,2,2,2],
        >>>                    [0.2,0.2,1.1,2,2,2,2,2]]).run().show()
        ————————————— pp: 8, vp: 3, micro: 16 ————————————
        --------------------  bubble  --------------------
        real    =   ideal   +   imba    +   comm    + recompute
        0.4444   =  0.1458   +  0.1851   +  0.0724   +  0.0412
        --------------------  memory  --------------------
        peak memory: 40.40, 43.60, 46.80, 50.00, 46.00, 42.00, 38.00, 34.00

        Show timeline without comm for vp=2, pp=15, micro=16, total 96 layers with extra equivalent 1.2 layer:
        >>> PipelineSimulator([[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
        >>>                    [3,3,3,3,3,3,3,3,4,4,4,4,4,4,3+1.2]], 16).run(False).show()
        ————————————— pp:15, vp: 2, micro: 16 ————————————
        --------------------  bubble  --------------------
        real    =   ideal   +   imba
        0.5741   =  0.4375   +  0.1366
        --------------------  memory  --------------------
        peak memory: 96.00, 96.00, 96.00, 96.00, 96.00, 96.00, 96.00, 93.00, 103.00, 97.00, 91.00,
        85.00, 79.00, 73.00, 70.20
    """
    def __init__(self, block_time: list, micro_num: int, *args, comm_time: float = 0.1,
                 layer_recompute=False, block_mem=1, block_mem_par=0, constant_mem=0,
                 backward_ratio=2., sub_fig=None, **kwargs):
        self.init(block_time, micro_num, comm_time, layer_recompute, block_mem,
                  block_mem_par, constant_mem, backward_ratio, sub_fig, *args, **kwargs)

    # pylint: disable=W0613
    def init(self, block_time, micro_num, comm_time,
             layer_recompute, block_mem, block_mem_par, constant_mem, backward_ratio,
             sub_fig, *args, **kwargs):
        r"""init"""
        self.micro_num = micro_num
        self.pp, self.vp = self._base_init(block_time)
        self.block_num = 2 * self.vp * self.micro_num
        self.comm_time = comm_time
        self._input_format(block_time, layer_recompute, block_mem, block_mem_par, backward_ratio)
        self.constant_mem = constant_mem
        self._statistic_init()
        self._comm = True
        self.adjust_func_list = [self.swap_send_rec]
        self.sub_fig = sub_fig
        # Construct pipeline blocks
        if self.vp == 1:
            method = '1f1b'
        else:
            method = kwargs.get('method', 'vpp')
            if self.micro_num >= self.pp:
                self.adjust_func_list = [self.vpp_send_delay, self.residue_delay] + self.adjust_func_list
        pp_builder = PipelineBuilder.get_builder(method)
        self.blocks = [pp_builder(self.pp, self.micro_num, self.vp, p, self.block_time[:, p],
                                  self.backward_time[:, p], self.block_mem[:, p], self.block_mem_par[:, p])
                       for p in range(self.pp)]

        self._build_block() # create connection among compute blocks
        self._build_comm_block() # create comm blocks for each compute block

    def run(self, comm=True, print_info=True) -> PipelineSimulator:
        r"""
        Run simulation to scheduler the pipeline.

        Args:
            comm(bool): whether build the pipeline considering communication dependency and time. Default: True.
            print_info(bool): whether automatically print bubble and memory information. Default: True.

        Raise:
            CasualError: the block sequences exist dependency loop.
            CausalCommError: the block with comm sequences exist dependency loop.

        Return:
            PipelineSimulator
        """
        self._comm = comm
        self._check_loop()
        if comm:
            self.lines = self._create_lines(*self.adjust_func_list)
            self._check_comm_loop()
            for b in range(self.block_num):
                for p in range(self.pp):
                    self.blocks[p][b].build_with_comm()
            self.lines[0][-1].build_with_comm()
        else:
            for p in range(self.pp):
                for block in self.blocks[p]:
                    block.build_without_comm()
        self._statistic_info()
        if print_info:
            self.print_info()
        return self

    def draw(self, comm=True, connect=None) -> PipelineSimulator:
        r"""
        Show the pipeline and memory timeline.

        Args:
            comm(bool): whether show the comm blocks. Default: True.
            connect(bool): whether show the connect arrow of send-receive pair when comm pipeline is built.
                Default: None.

        Return:
            PipelineSimulator
        """
        self.canvas = PlotMgr(2, ['block', 'memory'], sub_fig=self.sub_fig)
        if self._comm:
            connect = True if connect is None else connect
            self.canvas.draw(self.lines, 0, comm, connect, False, 'timeline')
        else:
            connect = False if connect is None else connect
            self.canvas.draw(self.blocks, 0, comm, connect, False, 'timeline')
        self.canvas.draw_mem(self.states.get('block_mem_list', []), 1)
        self.canvas.draw_info(self.bubbles, self.peak_memory)
        return self


    def show(self, comm=True, connect=None, file_name=None) -> PipelineSimulator:
        self.draw(comm, connect)
        self.canvas.show(file_name)
        return self

    def save(self, file_name, comm=True, connect=None) -> PipelineSimulator:
        self.draw(comm, connect)
        self.canvas.save(file_name)
        return self

    def print_info(self):
        r"""
        Print bubble and peak memory information.

        Return:
            PipelineSimulator
        """
        print('\033[1;37m' + '—' * 13, f'pp:{self.pp:>2}, vp:{self.vp:>2},',
              f'micro:{self.micro_num:>3} ' + '—' * 12 + '\033[0m')
        print('-' * 20, ' bubble ', '-' * 20)
        print(apply_format(apply_color(list(self.bubbles.keys()), ['1;33', '1;32', '1;31', '1;35', '1;36'])))
        print(apply_format(apply_color(list(self.bubbles.values()), ['1;33', '1;32', '1;31', '1;35', '1;36'])))
        print('-' * 20, ' memory ', '-' * 20)
        print(f"peak memory: {', '.join([f'{v:.2f}' for v in self.peak_memory])}")
        return self

    def _base_init(self, block_time) -> tuple:
        r"""init base setting"""
        if isinstance(block_time, (list, tuple)):
            if all(isinstance(item, (list, tuple)) for item in block_time):
                vp = len(block_time)
                pp = len(block_time[0])
            elif all(isinstance(item, (int, float)) for item in block_time):
                vp = 1
                pp = len(block_time)
            else:
                raise ValueError(f"Unsupported input format block_time: {block_time}")
        else:
            raise ValueError(f"Unsupported input format block_time: {block_time}")
        if self.micro_num < pp:
            raise ValueError(f" `micro_num`({self.micro_num}) should equal or larger than `pp`({pp})")
        return pp, vp

    def _input_format(self, block_time, layer_recompute, block_mem, block_mem_par, backward_ratio) -> None:
        r"""format inputs as 2d array"""
        self.block_time = format_2d_inputs(block_time, self.vp, self.pp)
        if isinstance(layer_recompute, bool):
            self.layer_recompute = self.block_time if layer_recompute else format_2d_inputs(0, self.vp, self.pp)
        else:
            self.layer_recompute = format_2d_inputs(layer_recompute, self.vp, self.pp)
        if isinstance(block_mem, (int, float)):
            self.block_mem = self.block_time * block_mem
        else:
            self.block_mem = format_2d_inputs(block_mem, self.vp, self.pp)

        if isinstance(block_mem_par, (int, float)):
            self.block_mem_par = self.block_time * block_mem_par
        else:
            self.block_mem_par = format_2d_inputs(block_mem_par, self.vp, self.pp)

        self.backward_ratio = format_2d_inputs(backward_ratio, self.vp, self.pp)

    def _statistic_init(self) -> None:
        r"""init statistic info"""
        self.forward_time = self.block_time
        self.backward_time = np.flip(self.block_time * self.backward_ratio + self.layer_recompute, axis=0)
        self.states = {'last_time': np.zeros(self.pp),
                       'warmup_time': np.zeros(self.pp),
                       'cooldown_time': np.zeros(self.pp),
                       'stable_free_time': (np.zeros((self.vp, self.pp)), np.zeros((self.vp, self.pp))),
                       'block_mem_list': [np.array([[0, 0]]) for _ in range(self.pp)]}
        self.model_compute_time = (np.sum(self.forward_time) + \
                                   np.sum(self.forward_time * self.backward_ratio)) * self.micro_num
        self.hardware_compute_time = (np.sum(self.forward_time) + np.sum(self.backward_time)) * self.micro_num
        self.bubbles = {'real': 0,
                        'ideal': (self.pp - 1) / self.vp / self.micro_num,
                        'imba': 0,
                        'comm': 0}
        if np.sum(self.layer_recompute) > 1e-5:
            self.bubbles['recompute'] = self.hardware_compute_time / self.model_compute_time - 1
        p, v, m = self.pp, self.vp, self.micro_num
        if self.vp == 1:
            if self.pp == 2:
                self.bubbles['comm'] = 4 * m
            elif self.pp % 2 == 0:
                self.bubbles['comm'] = 4 * p * m + 4 * p ** 2 - 14 * p
            else:
                self.bubbles['comm'] = 4 * p * m + 4 * p ** 2 - 12 * p
        elif self.pp <= 5:
            comm_coef_list = [[4, -2, 0], [6, -2, -6], [4, 0, 12], [6, -2, 40]]
            self.bubbles['comm'] = np.dot(np.array([p * v * m, m * p, 1]), comm_coef_list[self.pp - 2])
        elif self.pp % 2 == 0:
            self.bubbles['comm'] = 4 * p * v * m + 4 * p ** 2 - 13 * p
        else:
            self.bubbles['comm'] = 6 * p * v * m - 2 * v * p ** 2 + 4 * v * p - 2 * p * m + 6 * p ** 2 - 16 * p

        self.bubbles['comm'] *= self.comm_time / self.model_compute_time

    def _statistic_info(self) -> None:
        r"""compute statistic info"""
        for p in range(self.pp):
            blocks = self.lines[p] if self._comm else self.blocks[p]
            current_mem = self.constant_mem + blocks[0].mem_par

            for block in blocks:
                if block.type == 'c' and block.state == 'f':
                    current_mem += block.mem
                elif block.type == 'c' and block.state == 'b':
                    if not self._comm or not block.rec_block:
                        current_mem -= block.mem
                elif block.type == 'r' and block.host.state == 'b':
                    current_mem -= block.host.mem
                    block = block.host
                else:
                    continue
                self.states['block_mem_list'][p] = np.append(self.states['block_mem_list'][p],
                                                             np.array([[block.end, current_mem]]), axis=0)
            self.states['block_mem_list'][p] = np.append(self.states['block_mem_list'][p],
                                                         np.array([[blocks[-1].end, current_mem]]), axis=0)
        self.peak_memory = [np.max((self.states['block_mem_list'][p].T)[1]) for p in range(self.pp)]
        self.end_time = max(np.max((self.states['block_mem_list'][p].T)[0]) for p in range(self.pp))
        self.bubbles['real'] = (self.pp * self.end_time - self.model_compute_time) / self.model_compute_time
        self.bubbles['imba'] = self.bubbles['real'] - self.bubbles['ideal'] + 1e-10
        if not self._comm:
            self.bubbles.pop('comm')
        else:
            self.bubbles['imba'] -= self.bubbles['comm']
        if self.bubbles.get('recompute'):
            self.bubbles['imba'] -= self.bubbles['recompute']

    def _get_pre_label(self, label: tuple) -> tuple:
        r"""get pre block label"""
        t, s, m, v, p = label
        if (s, v, p) == ('f', 0, 0):
            return ('h', p)
        if (s, p) == ('f', 0):
            res = (t, s, m, v - 1, self.pp - 1)
            return res
        if (s, p) == ('b', self.pp - 1):
            if v == 0:
                res = (t, 'f', m, self.vp - 1, p)
                return res
            res = (t, s, m, v - 1, 0)
            return res
        if s == 'f':
            res = (t, s, m, v, p - 1)
            return res
        if s == 'b':
            res = (t, s, m, v, p + 1)
            return res
        raise ValueError(f"Illegal label: {label}")

    def _build_block(self) -> None:
        r"""Build `pre` relation for computation blocks."""
        books = {self.blocks[0][0].pre.label: self.blocks[0][0].pre}
        for p in range(self.pp):
            for item in self.blocks[p]:
                books[item.label] = item
        for p in range(self.pp):
            block = self.blocks[p][0]
            while block is not None:
                pre_label = self._get_pre_label(block.label)
                block.pre = books.get(pre_label, None)
                block = block.right

    def _build_comm_block(self) -> None:
        r"""Build `send_block` and `rec_block` relation among a computation block and two comm blocks."""
        for p in range(self.pp):
            block = self.blocks[p][0]
            while block is not None:
                pre = block.pre
                if pre.stage != block.stage:
                    block.rec_block = RecBlockSim(p, block.state, block.id, block.chunk, self.comm_time)
                    pre.send_block = SendBlockSim(pre.stage, pre.state, pre.id, pre.chunk, self.comm_time)
                    block.rec_block.host = block
                    block.rec_block.dual = pre.send_block
                    pre.send_block.host = pre
                    pre.send_block.dual = block.rec_block
                    block.depend_pre = block.rec_block
                    block.rec_block.depend_pre = pre.send_block
                    pre.send_block.depend_pre = pre
                else:
                    block.depend_pre = pre
                block = block.right

    def _check_loop(self) -> None:
        r"""check the existence of dependency"""
        loop = self.blocks[0][-1].loop()
        if loop:
            raise CausalError('Block dependency exist loops!', self.blocks, loop)
        for p in range(self.pp):
            for block in self.blocks[p]:
                block.flag = False

    def _check_comm_loop(self) -> None:
        r"""check the existence of comm dependency"""
        loop = self.lines[0][-1].comm_loop()
        if loop:
            raise CausalCommError('Block comm dependency exist loops!', self.lines, loop)
        for p in range(self.pp):
            for block in self.lines[p]:
                block.flag = False

    def _create_lines(self, *adjust_func) -> list[list[BlockSim]]:
        r"""create block line for each stage with comm"""
        lines = [copy.copy(self.blocks[p]) for p in range(self.pp)]
        for p in range(self.pp):
            for b in range(self.block_num):
                block = self.blocks[p][b]
                pre = block.pre
                if block.rec_block:
                    lines[p].insert(lines[p].index(block), block.rec_block)
                    if pre.type == 'h':
                        lines[pre.stage].insert(0, pre.send_block)
                    else:
                        lines[pre.stage].insert(lines[pre.stage].index(pre) + 1, pre.send_block)
        for func in adjust_func:
            lines = func(lines)
        for p in range(self.pp):
            for b, block in enumerate(lines[p]):
                if b == 0:
                    block.depend_left = block.left if block.left else block.host.left
                else:
                    block.depend_left = lines[p][b - 1]
        return lines

    def _get_block_phase(self, p: int, b: int) -> str:
        r"""get block phase"""
        r = self.micro_num % self.pp
        if b < (self.vp + 1) * self.pp - 2 * p - 2 + r:
            return 'warmup'
        if b > self.block_num - (self.vp + 1) * self.pp + 2 * p:
            return 'cooldown'
        return 'stable'

    def _send_block_delay(self, lines, p: int, b: int, distance: int) -> None:
        r"""adjust send block: delay send block"""
        i_send = lines[p].index(self.blocks[p][b].send_block)
        send_block = lines[p].pop(i_send)
        i_new = lines[p].index(self.blocks[p][b + distance]) + 1
        lines[p].insert(i_new, send_block)

    def _process_swap(self, block, lines, p, b, i_b, i_bn) -> bool:
        r"""process swap in condition"""
        if i_bn - i_b == 3:
            if p % 2 == 0 and lines[p][i_b + 1].type == 'r' and lines[p][i_b + 2].type == 's':
                lines[p][i_b + 1], lines[p][i_b + 2] = lines[p][i_b + 2], lines[p][i_b + 1]
            if p % 2 == 1 and lines[p][i_b + 1].type == 's' and lines[p][i_b + 2].type == 'r':
                if block.phase == 'warmup' and self.blocks[p][b + 1].phase == 'cooldown':
                    return False
                lines[p][i_b + 1], lines[p][i_b + 2] = lines[p][i_b + 2], lines[p][i_b + 1]
            if lines[p][i_b + 1].dual.stage == lines[p][i_b + 2].dual.stage:
                pd = lines[p][i_b + 1].dual.stage
                j_b1 = lines[pd].index(lines[p][i_b + 1].dual)
                j_b2 = lines[pd].index(lines[p][i_b + 2].dual)
                if j_b1 > j_b2:
                    lines[p][i_b + 1], lines[p][i_b + 2] = lines[p][i_b + 2], lines[p][i_b + 1]
        if i_bn - i_b == 4:
            if lines[p][i_b + 1].dual.stage == lines[p][i_b + 2].dual.stage and \
                lines[p][i_b + 2].dual.stage == lines[p][i_b + 3].dual.stage:
                if lines[p][i_b + 1].type == 's' and lines[p][i_b + 2].type == 's' \
                    and lines[p][i_b + 3].type == 'r':
                    lines[p][i_b + 1], lines[p][i_b + 2] = lines[p][i_b + 2], lines[p][i_b + 1]
        return True

    def swap_send_rec(self, lines) -> list[list[BlockSim]]:
        r"""adjust send block: swap send and receive in condition"""
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if b >= len(self.blocks[p]) - 1:
                    continue
                i_b = lines[p].index(block)
                i_bn = lines[p].index(self.blocks[p][b + 1])
                self._process_swap(block, lines, p, b, i_b, i_bn)
        return lines

    def vpp_send_delay(self, lines) -> list[list[BlockSim]]:
        r"""adjust send block: delay in stable phase"""
        if self.micro_num % self.pp != 0:
            return lines
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if block.send_block is not None and block.phase == 'stable':
                    self._send_block_delay(lines, p, b, 1)
        return lines

    def residue_delay(self, lines) -> list[list[BlockSim]]:
        r"""adjust send block: delay for residue"""
        r = self.micro_num % self.pp
        if r == 0:
            return lines
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if block.send_block is None:
                    continue
                if p == self.pp - 1 and block.id < self.pp + r and block.state == 'f':
                    self._send_block_delay(lines, -1, b, r + max(0, block.id - self.pp + 1))
                elif p == 0 and block.id < self.pp + r and block.state == 'b':
                    if self.micro_num // self.pp == 1:
                        self._send_block_delay(lines, 0, b, r)
                    else:
                        self._send_block_delay(lines, 0, b, r + self.pp)
                elif block.phase == 'stable':
                    self._send_block_delay(lines, p, b, 1)
        return lines


if __name__ == '__main__':

    PipelineSimulator([[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4 + 0.8]], 8, 0.1,
                      [[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]],
                      [[1.1, 2, 2, 2], [1.1, 2, 2, 2], [1.1, 1.1, 2, 2]], method='vpp').run().show()
