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
"""Pipeline parallel net"""

from collections import OrderedDict
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, mint
from mindspore.nn import Cell
from mindspore.common.initializer import initializer, HeUniform
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.experimental.parallel_core.pynative.transformer import get_attention_mask
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.transformer import _get_num_layers
from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import VocabParallelEmbedding
from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_world_size


class FakeData:
    """ generate fake data for pipeline parallel test """
    def __init__(self, data_num, seq_length, input_data=None):
        super().__init__()
        if input_data is not None:
            self.input_data = input_data
            self.data_num = self.input_data.shape[0]
            self.seq_length = self.input_data[0].shape[0]
        else:
            self.input_data = np.random.randint(0, 100, (data_num, seq_length))
        self.labels = np.random.randint(0, 100, (data_num, seq_length))

    def __getitem__(self, index):
        return Tensor(self.input_data[index], dtype=ms.int32), Tensor(self.labels[index], dtype=ms.int32)

    def __len__(self):
        return self.input_data.shape[0]


class FakeTransformerLayer(Cell):
    """ fake transformer layer """
    def __init__(self, seq_length, hidden_size, param_init='he_uniform'):
        super().__init__()
        self.first_liner = nn.Dense(in_channels=hidden_size,
                                    out_channels=seq_length,
                                    has_bias=False,
                                    weight_init=param_init)
        self.second_liner = nn.Dense(in_channels=seq_length,
                                     out_channels=hidden_size,
                                     has_bias=False,
                                     weight_init=param_init)

    def construct(self, hidden_state, attention_mask):
        """ fake transformer layer forward """
        hidden_state = self.first_liner(hidden_state)
        if attention_mask is not None:
            hidden_state *= attention_mask
        hidden_state = self.second_liner(hidden_state)
        return hidden_state


class FakeTransformer(Cell):
    """ fake transformer """
    def __init__(self,
                 config,
                 batch_size,
                 seq_length,
                 hidden_size,
                 param_init='he_uniform',
                 pre_process=False):
        super().__init__()
        self.pre_process = pre_process
        self.num_layers, self.offset = _get_num_layers(config, None)
        layers_dict = OrderedDict()
        for i in range(self.num_layers):
            layers_dict[str(i + self.offset)] = FakeTransformerLayer(seq_length,
                                                                     hidden_size,
                                                                     param_init=param_init)
        self.fake_transformer_layers = nn.SequentialCell(layers_dict)
        self.pipeline_parallel = get_pipeline_model_parallel_world_size() > 1
        if self.pipeline_parallel:
            self.set_hidden_states = ms.Parameter(P.zeros((batch_size, seq_length, hidden_size),
                                                          dtype=ms.float32),
                                                  requires_grad=False,
                                                  name="set_hidden_states")

    def set_input_tensor(self, input_tensor):
        """ set input """
        self.set_hidden_states.set_data(input_tensor, slice_shape=True)

    def construct(self, hidden_states, attention_mask):
        """ fake transformer forward """
        if not self.pre_process and self.pipeline_parallel:
            hidden_states = self.set_hidden_states

        for i in range(len(self.fake_transformer_layers)):
            hidden_states = self.fake_transformer_layers[i](hidden_states, attention_mask)
        return hidden_states


class FakeHead(Cell):
    """ fake head layer """
    def __init__(self, hidden_size, vocab_size, param_init='he_uniform', share_embedding_weight=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.matmul = P.MatMul(transpose_b=True)
        if share_embedding_weight and get_pipeline_model_parallel_world_size() > 1:
            param_init = "zeros"
            self.weight = ms.Parameter(initializer(init=param_init,
                                                   shape=(self.vocab_size, self.hidden_size),
                                                   dtype=ms.float32), name="weight")
        elif not share_embedding_weight:
            self.weight = ms.Parameter(initializer(init=param_init,
                                                   shape=(self.vocab_size, self.hidden_size),
                                                   dtype=ms.float32), name="weight")
        else:
            self.weight = None

    def construct(self, hidden_state, weight=None):
        """ fake head layer forward """
        if self.weight is not None:
            embedding_table = self.weight
        elif weight is not None:
            embedding_table = weight
        else:
            raise RuntimeError("Head layer's weight is not initialized, and input 'weight' is None")
        hidden_state = hidden_state.reshape((-1, self.hidden_size))
        logits = self.matmul(hidden_state, embedding_table)
        return logits


class FinalLossLayer(Cell):
    """ final loss layer """
    def __init__(self):
        super().__init__()
        self.entropy = CrossEntropyLoss()

    def construct(self, logits, labels):
        """ loss layer forward """
        input_mask = P.ones_like(labels).reshape((-1,))
        labels = labels.reshape((-1,))
        loss = self.entropy(logits, labels, input_mask)
        return loss[0]


class PipelineTestNet(Module):
    """ test net for pipeline parallel """
    def __init__(self, config, pre_process=True, post_process=True):
        super().__init__(config)
        self.num_layers = config.num_layers
        self.compute_dtype = config.compute_dtype
        hidden_size = config.hidden_size
        seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        batch_size = config.dataset_config.batch_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.pipeline_parallel = get_pipeline_model_parallel_world_size() > 1
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights

        if self.pre_process:
            self.embedding = VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=hidden_size,
                init_method=HeUniform(),
                reduce_scatter_embeddings=config.parallel_config.sequence_parallel,
                config=config,
                param_init_dtype=config.params_dtype
            )
            self.embedding.shared = True
            self.embedding.shared_embedding = True
            self.shared_weight_name_list.append(self.embedding.weight.name)

        self.fake_transformer = FakeTransformer(config,
                                                batch_size,
                                                seq_length,
                                                hidden_size,
                                                pre_process=self.pre_process)

        if self.post_process:
            self.fake_head = FakeHead(hidden_size,
                                      self.vocab_size,
                                      share_embedding_weight=not self.untie_embeddings_and_output_weights)
            if self.pipeline_parallel:
                self.fake_head.shared = True
                self.fake_head.shared_embedding = True
                self.shared_weight_name_list.append(self.fake_head.weight.name)
            self.final_norm = nn.LayerNorm((hidden_size,))
            self.total_loss = FinalLossLayer()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.fake_transformer.set_input_tensor(input_tensor)

    def construct(self, input_ids, labels=None, attention_mask=None):
        """ pipeline test net forward """
        if attention_mask is None:
            input_mask = mint.ne(input_ids, self.vocab_size + 1).astype(
                self.compute_dtype
            )
            attention_mask = get_attention_mask(input_mask)
        hidden_states = None
        # embedding
        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        # transformer
        hidden_states = self.fake_transformer(hidden_states, attention_mask)
        if self.post_process:
            # final norm
            hidden_states = self.final_norm(hidden_states)
            if self.untie_embeddings_and_output_weights:
                embedding_table = None
            else:
                if not self.pipeline_parallel:
                    embedding_table = self.embedding.weight
                else:
                    embedding_table = None
            # head
            logits = self.fake_head(hidden_states, embedding_table)
            # loss
            labels = labels.reshape((-1,))
            loss = self.total_loss(logits, labels)
            return loss
        return hidden_states
