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
"""mindformers parallel core"""
from mindformers.core.context import get_context
from mindformers.experimental.graph.tensor_parallel.layers import (RowParallelLinear as GraphRPLinear,
                                                                   ColumnParallelLinear as GraphCPLinear,
                                                                   VocabParallelEmbedding as GraphVPEmbedding)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.\
    layers import (RowParallelLinear as PynativeRPLinear,
                   ColumnParallelLinear as PynativeCPLinear,
                   VocabParallelEmbedding as PynativeVPEmbedding)
from mindformers.experimental.graph.transformer.\
    rotary_pos_embedding import (RotaryEmbedding as GraphRotaryEmbedding,
                                 apply_rotary_pos_emb as graph_apply_rotary_pos_emb)
from mindformers.experimental.parallel_core.pynative.transformer.\
    rotary_pos_embedding import (RotaryEmbedding as PynativeRotaryEmbedding,
                                 apply_rotary_pos_emb as pynative_apply_rotary_pos_emb)
from mindformers.experimental.graph.transformer.\
    transformer import (ParallelTransformer as GraphParallelTransformer,
                        ParallelTransformerLayer as GraphParallelTransformerLayer,
                        ParallelAttention as GraphParallelAttention,
                        ParallelMLP as GraphParallelMLP)
from mindformers.experimental.parallel_core.pynative.transformer.\
    transformer import (ParallelTransformer as PynativeParallelTransformer,
                        ParallelTransformerLayer as PynativeParallelTransformerLayer,
                        ParallelAttention as PynativeParallelAttention,
                        ParallelMLP as PynativeParallelMLP)
from mindformers.experimental.graph.optimizer.adamw import AdamW as GraphAdamW
from mindformers.experimental.parallel_core.pynative.optimizer.zero.adamw_zero import AdamW as PynativeAdamW
from mindformers.experimental.graph.transformer.language_model import get_language_model as graph_get_language_model
from mindformers.experimental.parallel_core.pynative.transformer.\
    language_model import (get_language_model as pynative_get_language_model)
from mindformers.experimental.graph.transformer.language_model import Embedding as GraphEmbedding
from mindformers.experimental.parallel_core.pynative.transformer.language_model import Embedding as PynativeEmbedding
from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry

__all__ = [
    'RowParallelLinear',
    'ColumnParallelLinear',
    'VocabParallelEmbedding',
    'RotaryEmbedding',
    'apply_rotary_pos_emb',
    'ParallelTransformer',
    'ParallelTransformerLayer',
    'ParallelAttention',
    'ParallelMLP',
    'AdamW',
    'get_language_model',
    'Embedding'
]


class RowParallelLinear:
    """Row parallel linear router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphRPLinear(*args, **kwargs)
        if mode == 1:
            return PynativeRPLinear(*args, **kwargs)
        return None


class ColumnParallelLinear:
    """Column parallel linear router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphCPLinear(*args, **kwargs)
        if mode == 1:
            return PynativeCPLinear(*args, **kwargs)
        return None


class VocabParallelEmbedding:
    """Vocab parallel embedding router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphVPEmbedding(*args, **kwargs)
        if mode == 1:
            return PynativeVPEmbedding(*args, **kwargs)
        return None


class RotaryEmbedding:
    """Rotary embedding router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphRotaryEmbedding(*args, **kwargs)
        if mode == 1:
            return PynativeRotaryEmbedding(*args, **kwargs)
        return None


def apply_rotary_pos_emb(*args, **kwargs):
    """Apply rotary position embedding router for graph or pynative depending on mode."""
    mode = get_context('mode')
    if mode == 0:
        return graph_apply_rotary_pos_emb(*args, **kwargs)
    if mode == 1:
        return pynative_apply_rotary_pos_emb(*args, **kwargs)
    return None


class ParallelTransformer:
    """Parallel transformer router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphParallelTransformer(*args, **kwargs)
        if mode == 1:
            return PynativeParallelTransformer(*args, **kwargs)
        return None


class ParallelTransformerLayer:
    """Parallel transformer layer router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphParallelTransformerLayer(*args, **kwargs)
        if mode == 1:
            return PynativeParallelTransformerLayer(*args, **kwargs)
        return None


class ParallelAttention:
    """Parallel attention router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphParallelAttention(*args, **kwargs)
        if mode == 1:
            return PynativeParallelAttention(*args, **kwargs)
        return None


class ParallelMLP:
    """Parallel MLP router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphParallelMLP(*args, **kwargs)
        if mode == 1:
            return PynativeParallelMLP(*args, **kwargs)
        return None


@ModuleRegistry.register_decorator(ModuleType.OPTIMIZER, "experiment_adamw")
class AdamW:
    """AdamW optimizer router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphAdamW(*args, **kwargs)
        if mode == 1:
            return PynativeAdamW(*args, **kwargs)
        return None


def get_language_model(*args, **kwargs):
    """Get language model router for graph or pynative depending on mode."""
    mode = get_context('mode')
    if mode == 0:
        return graph_get_language_model(*args, **kwargs)
    if mode == 1:
        return pynative_get_language_model(*args, **kwargs)
    return None


class Embedding:
    """Embedding router for graph or pynative depending on mode."""
    def __new__(cls, *args, **kwargs):
        mode = get_context('mode')
        if mode == 0:
            return GraphEmbedding(*args, **kwargs)
        if mode == 1:
            return PynativeEmbedding(*args, **kwargs)
        return None
