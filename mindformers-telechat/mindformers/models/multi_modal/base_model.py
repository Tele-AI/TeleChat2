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

"""
BaseXModalToTextModel
"""
import abc

from mindspore import Tensor
from mindspore import ops
from mindspore.ops import operations as P

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.modeling_utils import PreTrainedModel


class BaseXModalToTextModel(PreTrainedModel):
    """Base modal of multi-modal to text """
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.tensor_scatter_update = ops.TensorScatterUpdate().shard(((1, 1, 1),
                                                                      (1, 1, 1),
                                                                      (1, 1, 1)))

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        raise NotImplementedError("this method needs to be implemented in multi-card inference")

    def update_modal_to_text(self, modal_embeds: Tensor, text_embeds: Tensor, modal_context_pos: Tensor):
        """update the value at a specific position of the text embedding with the modal embeddings"""
        modal_embeds = self.cast(modal_embeds, text_embeds.dtype)
        text_embeds = self.tensor_scatter_update(text_embeds, modal_context_pos, modal_embeds)
        return text_embeds

    @abc.abstractmethod
    def kvcache(self, layer_idx):
        raise NotImplementedError("this method needs to be implemented in kbk inference")

    @abc.abstractmethod
    def set_dynamic_inputs(self, **kwargs):
        raise NotImplementedError("this method needs to be implemented in kbk inference and is_dynamic=True")
