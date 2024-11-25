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
"""InternLM models' APIs."""
import copy

from mindspore import nn, ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
import mindspore.common.dtype as mstype

from mindformers import Linear, CrossEntropyLoss
from mindformers.models import LlamaModel, LlamaForCausalLM
from mindformers.models.utils import LayerSetting
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.models.utils import lazy_inline

from internlm_transformer import InternLMDecodeLayer
from internlm_config import InternLMConfig


class InternLMModel(LlamaModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLMDecoderLayer`].

    Args:
        config(InternLMConfig): The config of network.

    """

    def __init__(self, config: InternLMConfig):
        super().__init__(config)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type,
                                             parallel_optimizer=True)
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.tok_embeddings.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = InternLMDecodeLayer(layer_id=layer_id,
                                        dim=config.hidden_size,
                                        n_heads=config.num_heads,
                                        n_kv_heads=config.n_kv_heads,
                                        intermediate_size=config.intermediate_size,
                                        multiple_of=config.multiple_of,
                                        ffn_dim_multiplier=config.ffn_dim_multiplier,
                                        norm_eps=config.rms_norm_eps,
                                        compute_dtype=config.compute_dtype,
                                        layernorm_compute_dtype=config.layernorm_compute_type,
                                        softmax_compute_dtype=config.softmax_compute_type,
                                        rotary_dtype=config.rotary_dtype,
                                        param_init_type=config.param_init_type,
                                        has_bias=config.has_bias,
                                        use_past=config.use_past,
                                        use_flash_attention=config.use_flash_attention,
                                        block_size=config.block_size,
                                        num_blocks=config.num_blocks,
                                        is_dynamic=config.is_dynamic,
                                        use_rope_slice=config.use_rope_slice,
                                        parallel_config=config.parallel_config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InternLMForCausalLM(LlamaForCausalLM):
    """Provide InternLM training loss or logits through network, inherited from [`LlamaForCausalLM`].

    Args:
        config(InternLMConfig): The config of network.

    """

    config_class = InternLMConfig
    base_model_prefix = "internlm"

    @lazy_inline
    def __init__(self, config: InternLMConfig):
        checkpoint_name_or_path = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = ""
        super().__init__(config)
        self.model = InternLMModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=mstype.float16,
                              skip_redistribution=config.is_dynamic,
                              weight_init="normal")
        vocab_size = config.vocab_size
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((1, 1), (dp * mp, 1)))
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        loss_parallel_config.model_parallel = dp * mp
        loss_parallel_config.data_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)
        config.checkpoint_name_or_path = checkpoint_name_or_path
        self.load_checkpoint(config)
