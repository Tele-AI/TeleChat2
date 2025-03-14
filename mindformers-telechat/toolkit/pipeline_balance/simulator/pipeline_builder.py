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
r"""Build pipeline scheduler"""
from __future__ import annotations
from toolkit.pipeline_balance.simulator.sim_block import MicroBlockSim, BlockSim, HeadBlockSim


class PipelineBuilder:
    r"""Build pipeline scheduler"""
    @staticmethod
    def _inter_merge(a: list[MicroBlockSim], b: list[MicroBlockSim], delta: int = 0) -> list[MicroBlockSim]:
        r"""merge forward and backward chain for 1f1b"""
        res = []
        if delta >= 0:
            res.extend(a[:delta])
            a = a[delta:]
        else:
            res.extend(b[:-delta])
            b = b[-delta:]
        stable_count = 0
        while a:
            block = a.pop(0)
            block.phase = 'stable'
            res.append(block)
            stable_count += 1
            if b:
                block = b.pop(0)
                block.phase = 'stable'
                res.append(block)
                stable_count += 1
            else:
                break
        if stable_count:
            res[-1].phase = 'cooldown'
        if a:
            res.extend(a)
        elif b:
            res.extend(b)
        return res

    @staticmethod
    def _build_chain(line: list[MicroBlockSim], p: int) -> list[BlockSim]:
        r"""build pipeline chain"""
        # pylint: disable=E1120
        head = HeadBlockSim(p)
        left = head
        for item in line:
            left.right = item
            item.left = left
            left = item
        if p == 0:
            head.right.pre = head
        return line

    @staticmethod
    # pylint: disable=W0613
    def build_1f1b(pp, micro_num, vp, p, forward_time, backward_time, block_mem, block_mem_par) -> list[BlockSim]:
        r"""1f1b pipeline"""
        forward_time = forward_time[0]
        backward_time = backward_time[0]
        block_mem = block_mem[0]
        block_mem_par = block_mem_par[0]
        for_line = [MicroBlockSim(p, 'f', i, 0, forward_time, mem=block_mem, mem_par=block_mem_par, phase='warmup')
                    for i in range(micro_num)]
        back_line = [MicroBlockSim(p, 'b', i, 0, backward_time, mem=block_mem, mem_par=block_mem_par, phase='cooldown')
                     for i in range(micro_num)]
        line = PipelineBuilder._inter_merge(for_line, back_line, pp - p - 1)
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def build_virtualpipeline(pp, micro_num, vp, p, forward_time,
                              backward_time, block_mem, block_mem_par) -> list[BlockSim]:
        r"""1f1b virtual pipeline"""
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i],
                                                   mem=block_mem[i], mem_par=block_mem_par[i],
                                                   phase='warmup') for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, i, backward_time[i],
                                                    mem=block_mem[vp - 1 - i], mem_par=block_mem_par[vp - 1 - i],
                                                    phase='cooldown') for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i],
                                               mem=block_mem[i], mem_par=block_mem_par[i],
                                               phase='warmup') for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, i, backward_time[i],
                                                mem=block_mem[vp - 1 - i], mem_par=block_mem_par[vp - 1 - i],
                                                phase='cooldown') for m in range(pp)])
        line = PipelineBuilder._inter_merge(for_line, back_line, (vp + 1) * pp - 2 * p - 2 + r * (vp - 1))
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def build_virtualpipeline2(pp, micro_num, vp, p, forward_time,
                               backward_time, block_mem, block_mem_par):
        r"""virtual pipeline with less memory scheduler"""
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i],
                                                   mem=block_mem[i], mem_par=block_mem_par[i],
                                                   phase='warmup') for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, i, backward_time[i],
                                                    mem=block_mem[vp - 1 - i], mem_par=block_mem_par[vp - 1 - i],
                                                    phase='cooldown') for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i],
                                               mem=block_mem[i], mem_par=block_mem_par[i],
                                               phase='warmup') for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, i, backward_time[i],
                                                mem=block_mem[vp - 1 - i], mem_par=block_mem_par[vp - 1 - i],
                                                phase='cooldown') for m in range(pp)])

        line = PipelineBuilder._inter_merge(for_line, back_line, vp * pp - p - 1)
        return PipelineBuilder._build_chain(line, p)

    @staticmethod
    def get_builder(method='1f1b'):
        r"""get pipeline builder"""
        if method == '1f1b':
            return PipelineBuilder.build_1f1b
        if method == 'vpp':
            return PipelineBuilder.build_virtualpipeline
        if method == 'vpp2':
            return PipelineBuilder.build_virtualpipeline2
        raise ValueError(f"`method` only support ['1f1b', 'vpp', 'vpp2'], but got {method}")
