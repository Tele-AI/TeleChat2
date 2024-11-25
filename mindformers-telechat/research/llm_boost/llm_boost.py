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
from mindspore.experimental.llm_boost.register import LlmBoostRegister

from research.llm_boost.llm_boost_config import LlmBoostConfig
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
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
        llm_boost_kwargs = {"config": config}
        self.llm_boost = LlmBoostRegister.get_instance(
            config.llm_backend, config.boost_model_name, **llm_boost_kwargs
        )
        self.llm_boost.init()
        self.is_set_kvcache = False
        self.parm_dict = {}

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation"""
        model_inputs = {}
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        if hasattr(self, "llm_boost"):
            batch_valid_length = kwargs.get("valid_length_each_example")
            block_tables = kwargs.get("block_tables")
            slot_mapping = kwargs.get("slot_mapping")
            prefill = kwargs.get("prefill")
            bs = batch_valid_length.shape[0]
            position_ids_list = [
                np.arange(context_len, dtype=np.int64)
                for context_len in batch_valid_length
            ]
            if input_ids.shape[-1] == 1:
                input_ids = np.concatenate(input_ids, 0)
            else:
                input_ids_list = []
                for i in range(bs):
                    context_len = batch_valid_length[i]
                    if prefill:
                        input_ids_list.append(input_ids[i][:context_len])
                    else:
                        input_ids_list.append(
                            input_ids[i][context_len - 1 : context_len]
                        )
                input_ids = np.concatenate(input_ids_list, 0)
            position_ids = np.concatenate(position_ids_list, 0)
            slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
            lm_head_indices = np.cumsum(batch_valid_length, dtype=np.int64) - 1
            seq_lens = batch_valid_length.tolist()
            model_inputs["llm_boost_inputs"] = {
                "input_ids": Tensor.from_numpy(input_ids),
                "position_ids": Tensor.from_numpy(position_ids),
                "lm_head_indices": Tensor.from_numpy(lm_head_indices),
                "block_tables": Tensor.from_numpy(block_tables),
                "slot_mapping": Tensor.from_numpy(slot_mapping),
                "batch_valid_length": Tensor.from_numpy(batch_valid_length),
                "seq_lens": seq_lens,
            }
        return model_inputs

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        pass

    def set_dynamic_inputs(self, **kwargs):
        pass

    # pylint: disable=W0613, C0330
    def construct(self, llm_boost_inputs=None, **kwargs):
        """llm boost forward"""
        if not self.is_set_kvcache:
            self.llm_boost.set_kvcache()
            self.is_set_kvcache = True
        self.llm_boost.add_flags(is_first_iteration=self.is_first_iteration)
        cos_embed, sin_embed, _ = self.freqs_mgr.prefill_flatten()
        llm_boost_inputs["cos_embed"] = cos_embed
        llm_boost_inputs["sin_embed"] = sin_embed
        return self.llm_boost.forward(llm_boost_inputs)
