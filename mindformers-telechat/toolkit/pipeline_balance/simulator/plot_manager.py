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
"""PlotMgr"""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

from toolkit.pipeline_balance.simulator.sim_block import MicroBlockSim, BlockSim


class PlotMgr:
    r"""Plot Manager"""
    # pylint: disable=W0613
    def __init__(self, *args, num_plots=2, ax_type='block', subplot_args=None,
                 sub_fig=None, **kwargs):
        if sub_fig:
            self.fig = sub_fig
        else:
            self.fig = plt.figure(figsize=kwargs.get('figsize', (12, 8)))
        self.fig.subplots_adjust(wspace=0, hspace=0.4)
        ax_type = ax_type if isinstance(ax_type, (list, tuple)) else [ax_type] * num_plots
        self.ax = []
        for i in range(num_plots):
            if subplot_args is None:
                self.ax.append(self.fig.add_subplot(num_plots * 100 + 10 + i + 1))
            elif isinstance(subplot_args, Iterable) and len(subplot_args) >= num_plots:
                self.ax.append(self.fig.add_subplot(subplot_args[i]))
            else:
                raise ValueError(f"Unsupported subplot_args format: {subplot_args}")

    def _set_block_ax(self, ax: plt.Axes, pp: int) -> plt.Axes:
        r"""set axis"""
        ax.set_title("Pipeline Flow Timeline")
        ax.set_yticks(range(pp), [f"stage {p}" for p in range(pp)])
        for tick in ax.get_yticklabels():
            tick.set_verticalalignment('top')
            tick.set_transform(tick.get_transform() + ScaledTranslation(0, 0.05 - 1 / pp, self.fig.dpi_scale_trans))
            tick.set_fontsize(12)
        ax.set_ylim(0, pp)
        ax.invert_yaxis()

    def _get_block_indices(self, blocks: list[list[MicroBlockSim]], mode='compact', equal_wide=False):
        r"""get block indices"""
        if mode not in ['compact', 'joint', 'timeline']:
            raise ValueError(f"Get unsupported draw mode: {mode}")
        if mode == 'timeline' and not blocks[-1][-1].finish:
            raise ValueError(f"Block building should be finished before drawing timeline")
        block_index = []
        for block_p in blocks:
            inds = []
            for block in block_p:
                if mode == 'compact':
                    if block.type == 'c':
                        inds.append(1 if equal_wide else block.time)
                    else:
                        inds.append(0)
                elif mode == 'joint':
                    if block.type == 'c':
                        inds.append(1 if equal_wide else block.time)
                    else:
                        inds.append(block.time)
                else:
                    inds.append(1)
            inds.insert(0, 0)
            inds = np.cumsum(inds)
            block_index.append(inds)
        return block_index

    def draw_block(self, block_index: list[list[float]], blocks: list[list[MicroBlockSim]],
                   ax_index: int = 0, equal_wide=False, width=1, phase=False):
        r"""draw compute blocks"""
        for p, block_p in enumerate(blocks):
            for b, block in enumerate(block_p):
                if block.type == 'c':
                    block.draw(self.ax[ax_index], index=block_index[p][b],
                               equal_wide=equal_wide, width=width, phase=phase)
        return self

    def draw_comm(self, block_index: list[list[float]], blocks: list[list[MicroBlockSim]],
                  ax_index: int = 0, equal_wide=False, mode='compact'):
        r"""draw comm blocks"""
        for p, block_p in enumerate(blocks):
            for b, block in enumerate(block_p):
                if block.type == 'c' and mode == 'compact':
                    if block.send_block:
                        block.send_block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide)
                    if block.rec_block:
                        block.rec_block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide)
                elif block.type in ['s', 'r'] and mode in ['joint', 'timeline']:
                    block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide, mode=mode)
        return self

    def draw_connect(self, block_index: list[list[float]], blocks: list[list[MicroBlockSim]],
                     ax_index: int = 0, equal_wide=False, mode='compact'):
        r"""draw connect arrow"""
        for p, block_p in enumerate(blocks):
            for b, block in enumerate(block_p):
                if block.type == 'c' and mode == 'compact' and block.send_block:
                    dual_p = block.send_block.dual.stage
                    dual_ind = blocks[dual_p].index(block.send_block.dual.host)
                    block.send_block.draw_comm(self.ax[ax_index], index_from=block_index[p][b],
                                               index_to=block_index[dual_p][dual_ind],
                                               equal_wide=equal_wide, mode=mode)
                elif block.type == 's' and mode in ['joint', 'timeline']:
                    dual_p = block.dual.stage
                    dual_ind = blocks[dual_p].index(block.dual)
                    block.draw_comm(self.ax[ax_index], index_from=block_index[p][b],
                                    index_to=block_index[dual_p][dual_ind], equal_wide=equal_wide, mode=mode)
        return self

    def draw(self, blocks: list[list[MicroBlockSim]], ax_index: int = 0,
             comm=False, connect=False, equal_wide=False, mode='compact', phase=False) -> PlotMgr:
        r"""draw pipeline timeline"""
        pp = len(blocks)
        block_index = self._get_block_indices(blocks, mode=mode, equal_wide=equal_wide)
        width = max(np.max(block_index[p]) for p in range(pp)) if blocks[0][-1].end is None \
            else max(blocks[p][-1].end for p in range(pp))
        self.draw_block(block_index, blocks, ax_index, equal_wide, width, phase=phase)
        if comm:
            self.draw_comm(block_index, blocks, ax_index, equal_wide, mode)
        if connect:
            self.draw_connect(block_index, blocks, ax_index, equal_wide, mode)
        self._set_block_ax(self.ax[ax_index], pp)
        self.ax[ax_index].set_xlim(0, width)
        self.ax[ax_index].set_xticks(np.linspace(0, width, 8))
        return self

    def draw_loop(self, blocks: list[list[MicroBlockSim]], loop: list[BlockSim],
                  ax_index: int = 0, comm=False, connect=False, equal_wide=False) -> PlotMgr:
        r"""draw dependency loop"""
        self.draw(blocks, ax_index, comm, connect, equal_wide, phase=True)
        block_index = self._get_block_indices(blocks, equal_wide=equal_wide)
        msg = 'dependency loop: '
        for b in range(len(loop) - 1):
            p = loop[b].stage
            ind = blocks[p].index(loop[b])
            x1, y1, dx1, _ = loop[b].loc_size(block_index[p][ind], equal_wide)
            p = loop[b + 1].stage
            ind = blocks[p].index(loop[b + 1])
            x2, y2, dx2, _ = loop[b + 1].loc_size(block_index[p][ind], equal_wide)
            msg = f'{msg} {loop[b].color_label} -> '
            self.ax[ax_index].annotate(None, xy=(x1 + dx1 / 2, y1), xytext=(x2 + dx2 / 2, y2),
                                       arrowprops=dict(fc='white', ec='r', arrowstyle='simple',
                                                       shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=-0.1"))
        self.msg = f'{msg} {loop[len(loop) - 1].color_label}'
        return self

    def draw_comm_loop(self, lines: list[list[BlockSim]], loop: list[BlockSim], ax_index: int = 0) -> PlotMgr:
        r"""draw comm dependency loop"""
        self.draw(lines, ax_index, True, True, True, 'joint', phase=True)
        block_index = self._get_block_indices(lines, mode='joint', equal_wide=True)
        msg = 'dependency loop: '
        for b in range(len(loop) - 1):
            p = loop[b].stage
            ind = lines[p].index(loop[b])
            x1, y1, dx1, _ = loop[b].loc_size(block_index[p][ind], True, 'joint')
            p = loop[b + 1].stage
            ind = lines[p].index(loop[b + 1])
            x2, y2, dx2, _ = loop[b + 1].loc_size(block_index[p][ind], True, 'joint')
            msg = f'{msg} {loop[b].color_label} -> '
            self.ax[ax_index].annotate(None, xy=(x1 + abs(dx1) / 2, y1), xytext=(x2 + abs(dx2) / 2, y2), size=10,
                                       arrowprops=dict(fc='white', ec='r', arrowstyle='simple',
                                                       shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=-0.1", lw=0.8))
        self.msg = f'{msg} {loop[len(loop) - 1].color_label}'
        return self

    def draw_mem(self, block_mem_list: list[np.ndarray], ax_index: int = 0) -> PlotMgr:
        for p, block_mem in enumerate(block_mem_list):
            self.ax[ax_index].plot((block_mem.T)[0], (block_mem.T)[1], label=f"stage-{p}")
        self.ax[ax_index].set_title("Block Memory Timeline")
        width = max(np.max((block_mem.T)[0]) for block_mem in block_mem_list)
        height = max(np.max((block_mem.T)[1]) for block_mem in block_mem_list)
        self.ax[ax_index].set_xlim(0, max(np.max((block_mem.T)[0]) for block_mem in block_mem_list))
        self.ax[ax_index].set_xticks(np.linspace(0, width, 8))
        self.ax[ax_index].set_yticks(np.linspace(0, height, 4))

    def draw_info(self, bubble_info: dict, mem_info: list):
        info_list = [f'{k} bubble: {v:.4f}' for k, v in bubble_info.items()]
        self.fig.text(0.5, 0.5, ', '.join(info_list), ha='center', va='center',
                      fontdict={'fontsize': 13, 'weight': 'medium'}, color='C3')
        info_list = [f"{v:.0f}" for v in mem_info]
        self.fig.text(0.5, 0.05, f"peak memory: {', '.join(info_list)}", ha='center', va='center',
                      fontdict={'fontsize': 10, 'weight': 'medium'}, color='C0')


    def save(self, file_name):
        self.fig.legend(bbox_to_anchor=(0.22, 0.45))
        plt.savefig(file_name)

    def show(self, file_name=None):
        self.fig.legend(bbox_to_anchor=(0.22, 0.45))
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()
