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
""" CausalError class"""
from __future__ import annotations
import matplotlib.pyplot as plt

from toolkit.pipeline_balance.simulator.sim_block import MicroBlockSim, BlockSim
from toolkit.pipeline_balance.simulator.plot_manager import PlotMgr


class CausalError(Exception):
    r""" CausalError without comm"""
    def __init__(self, msg, blocks: list[list[MicroBlockSim]], loop: list[BlockSim]) -> None:
        super().__init__()
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_loop(blocks, loop, 0, False, False, True)
        self.canvas.ax[0].set_title("Block pipeline dependency")
        print(f"{self.canvas.msg}")

    def __str__(self):
        plt.show()
        return f"{self.msg}"


class CausalCommError(Exception):
    r""" CausalError with comm"""
    def __init__(self, msg, blocks: list[list[MicroBlockSim]], loop: list[BlockSim]) -> None:
        super().__init__()
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_comm_loop(blocks, loop, 0)
        self.canvas.ax[0].set_title("Block comm pipeline dependency")
        print(f"{self.canvas.msg}")

    def __str__(self):
        plt.show()
        return f"{self.msg}"
