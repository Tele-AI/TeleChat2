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

"""mindformers init"""

from .mappings import all_to_all_hp2sp, all_to_all_sp2hp
from .mappings import CopyToModelParallelRegion
from .mappings import ScatterToModelParallelRegion
from .mappings import GatherFromModelParallelRegion
from .mappings import ReduceFromModelParallelRegion
from .mappings import ReduceScatterToSequenceParallelRegion
from .mappings import ReduceScatterToTensorParallelRegion
from .mappings import ScatterToSequenceParallelRegion
from .mappings import GatherFromSequenceParallelRegion
from .mappings import AllGatherFromTensorParallelRegion
from .mappings import GatherFromTensorAndExpertParallelRegion
from .mappings import AllToAll, AllToAllSP2HP
from .layers import ColumnParallelLinear, RowParallelLinear
from .layers import VocabParallelEmbedding
from .layers import LinearWithGradAccumulationAndAsyncCommunication
from .random import RNGStateTracer
from .random import get_rng_tracer, set_rng_seed
from .random import DATA_PARALLEL_GENERATOR
from .random import TENSOR_PARALLEL_GENERATOR
from .random import EXPERT_PARALLEL_GENERATOR
from .cross_entropy import VocabParallelCrossEntropy


__all__ = [
    'all_to_all_hp2sp',
    'all_to_all_sp2hp',
    'CopyToModelParallelRegion',
    'ScatterToModelParallelRegion',
    'GatherFromModelParallelRegion',
    'ReduceFromModelParallelRegion',
    'ReduceScatterToSequenceParallelRegion',
    'ReduceScatterToTensorParallelRegion',
    'ScatterToSequenceParallelRegion',
    'GatherFromSequenceParallelRegion',
    'AllGatherFromTensorParallelRegion',
    'GatherFromTensorAndExpertParallelRegion',
    'AllToAll',
    'AllToAllSP2HP',
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "LinearWithGradAccumulationAndAsyncCommunication",
    'RNGStateTracer',
    'get_rng_tracer',
    'set_rng_seed',
    'DATA_PARALLEL_GENERATOR',
    'TENSOR_PARALLEL_GENERATOR',
    'EXPERT_PARALLEL_GENERATOR',
    'VocabParallelCrossEntropy'
]
