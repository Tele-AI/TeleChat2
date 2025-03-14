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
"""Llm boost models' APIs."""
import numpy as np

from mindspore import Tensor
from mindspore.experimental.llm_boost.register import LlmBoostRegister, LlmBoostType
from research.llm_boost.llm_boost_config import LlmBoostConfig
from research.llm_boost.utils import is_support_lccl
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.version_control import need_nz
from mindformers.tools.utils import get_predict_run_mode
from mindformers.modules.layers import FreqsMgr


__all__ = ["LlmBoostForCausalLM"]


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlmBoostForCausalLM(PreTrainedModel):
    r"""
    Use third-party frameworks to accelerate large models
    Args:
        config (LlmBoostConfig): The config of llm boost model.

    Returns:
        output: Tensor, the output of llm decoderlayer

    """

    config_class = LlmBoostConfig

    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        self.use_past = config.use_past
        self.head_dim = config.hidden_size // config.num_heads
        self.is_first_iteration = True
        self.llm_backend = config.llm_backend
        self.predict_run_mode = get_predict_run_mode()
        self.freqs_mgr = FreqsMgr(
            head_dim=self.head_dim,
            seq_length=config.seq_length,
            max_position_embedding=config.max_position_embedding,
            rotary_dtype=config.rotary_dtype,
            theta=config.theta,
            scaling_factor=config.scaling_factor,
            extend_method=config.extend_method,
            parallel_config=config.parallel_config,
        )
        self.freqs_mgr.shard(config.parallel_config)
        config.need_nz = need_nz()
        if config.communication_backend == "":
            config.communication_backend = "lccl" if is_support_lccl() else "hccl"

        llm_boost_kwargs = {"config": config}
        self.llm_boost = LlmBoostRegister.get_instance(
            config.llm_backend, config.boost_model_name, **llm_boost_kwargs
        )
        self.llm_boost.init()
        self.is_set_kvcache = False
        self.need_prepare = (config.llm_backend == LlmBoostType.BUILDIN)

    # pylint: disable=C0111
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {}
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        batch_valid_length = kwargs.get("valid_length_each_example", None)
        position_ids = kwargs.get("position_ids", None)
        block_tables = kwargs.get("block_tables", None)
        slot_mapping = kwargs.get("slot_mapping", None)
        lm_head_indices = kwargs.get("prefill_head_indices", None)
        prefill = kwargs.get("prefill")

        if self.llm_backend == LlmBoostType.BUILDIN:
            model_inputs["llm_boost_inputs"] = self.prepare_inputs_for_build_in(
                prefill=prefill,
                input_ids=input_ids,
                position_ids=position_ids,
                batch_valid_length=batch_valid_length,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                lm_head_indices=lm_head_indices,
            )
        return model_inputs

    # pylint: disable=C0330, C0111
    def prepare_inputs_for_build_in(
        self,
        prefill,
        input_ids,
        position_ids,
        batch_valid_length,
        block_tables,
        slot_mapping,
        lm_head_indices,
    ):
        llm_boost_inputs = {}
        seq_lens = batch_valid_length.tolist()
        bs = batch_valid_length.shape[0]
        input_ids_list = []
        if prefill:
            if bs > 1 and input_ids.shape[0] != 1:
                for i in range(bs):
                    context_len = batch_valid_length[i]
                    input_ids_list.append(input_ids[i][:context_len])
                    slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))

            if position_ids is None:
                position_ids_list = [
                    np.arange(context_len, dtype=np.int64)
                    for context_len in batch_valid_length
                ]
                position_ids = np.concatenate(position_ids_list, 0)
            llm_boost_inputs["position_ids"] = Tensor.from_numpy(position_ids)
            if lm_head_indices is None:
                lm_head_indices = np.cumsum(batch_valid_length, dtype=np.int64) - 1
            llm_boost_inputs["lm_head_indices"] = Tensor.from_numpy(
                lm_head_indices.astype(np.int64)
            )
        else:
            if input_ids.shape[-1] != 1:
                for i in range(bs):
                    context_len = batch_valid_length[i]
                    input_ids_list.append(input_ids[i][context_len - 1 : context_len])
            if position_ids is None:
                position_ids = batch_valid_length - 1
        if input_ids_list:
            input_ids = np.concatenate(input_ids_list, 0)
        llm_boost_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int64))
        llm_boost_inputs["position_ids"] = Tensor.from_numpy(
            position_ids.astype(np.int64)
        )
        llm_boost_inputs["block_tables"] = Tensor.from_numpy(block_tables)
        llm_boost_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        llm_boost_inputs["batch_valid_length"] = Tensor.from_numpy(batch_valid_length)
        llm_boost_inputs["seq_lens"] = seq_lens
        return llm_boost_inputs

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return input_ids

    def set_dynamic_inputs(self, **kwargs):
        pass

    def add_flags_custom(self, is_first_iteration):
        self.is_first_iteration = is_first_iteration
        self.llm_boost.add_flags(is_first_iteration=self.is_first_iteration)

    # pylint: disable=W0613, C0330
    def construct(self, llm_boost_inputs=None, **kwargs):
        """llm boost forward"""
        if self.need_prepare:
            if not self.is_set_kvcache:
                self.llm_boost.set_kvcache()
                self.is_set_kvcache = True
            llm_boost_inputs["cos_embed"] = self.freqs_mgr.freqs_cos
            llm_boost_inputs["sin_embed"] = self.freqs_mgr.freqs_sin
            return self.llm_boost.forward(llm_boost_inputs)

        bvl = kwargs["batch_valid_length"]
        if bvl.ndim > 1:
            bvl = Tensor(bvl[0].asnumpy())
        return self.llm_boost.forward(kwargs["input_ids"], bvl, None)
