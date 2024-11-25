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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindformers.modules import AttentionMask
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import VocabParallelEmbedding
from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_world_size
from mindformers.experimental.parallel_core.pynative.utils import add_attr_for_shared_weight

class FakeData():
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


class PreprocessLayer(Module):
    """ preprocess layer """
    def __init__(self, config):
        super().__init__()
        self.pad_token = config.model_config.pad_token_id
        self.vocab_size = config.model_config.vocab_size
        self.seq_length = config.model_config.seq_length
        self.not_equel = P.NotEqual()
        self.cast = P.Cast()
        self.output_dict = {}
        self.get_attention_mask = AttentionMask(self.seq_length,
                                                compute_dtype=ms.float32)

    def construct(self, input_ids, labels=None, attention_mask=None):
        """ preprocess layer forward """
        if labels is None:
            input_ids, labels = input_ids[:, :self.seq_length], input_ids[:, 1:]
        if attention_mask is None:
            input_mask = self.cast(self.not_equel(input_ids, self.vocab_size + 1), ms.float32)
            attention_mask = self.get_attention_mask(input_mask)

        self.output_dict["input_ids"] = input_ids
        self.output_dict["attention_mask"] = attention_mask
        self.output_dict["labels"] = labels
        return self.output_dict


class FakeTransformerLayer(Module):
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


class FakeHead(Module):
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


class FinalLossLayer(Module):
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
    def __init__(self, config):
        super().__init__(config)
        num_layers = config.model_config.num_layers
        hidden_size = config.model_config.hidden_size
        seq_length = config.model_config.seq_length
        vocab_size = config.model_config.vocab_size

        self.preprocess = PreprocessLayer(config)
        self.embedding = VocabParallelEmbedding(num_embeddings=vocab_size,
                                                embedding_dim=hidden_size,
                                                config=config.parallel_config,
                                                init_method='he_uniform')
        add_attr_for_shared_weight(self.embedding)
        self.fake_transformer_layers = nn.SequentialCell()
        for _ in range(num_layers):
            self.fake_transformer_layers.append(FakeTransformerLayer(seq_length, hidden_size))

        self.fake_head = FakeHead(hidden_size,
                                  vocab_size,
                                  share_embedding_weight=self.share_embedding_weight)
        add_attr_for_shared_weight(self.fake_head)
        self.final_norm = nn.LayerNorm((hidden_size,))
        self.total_loss = FinalLossLayer()

        world_size = get_pipeline_model_parallel_world_size()
        if get_pipeline_model_parallel_world_size() > 1:
            self.preprocess.pipeline_stage = 0
            self.embedding.pipeline_stage = 0
            for idx in range(num_layers):
                stage = min(idx // (num_layers // world_size), world_size - 1)
                self.fake_transformer_layers[idx].pipeline_stage = stage
            self.fake_head.pipeline_stage = world_size - 1
            self.final_norm.pipeline_stage = world_size - 1
            self.total_loss.pipeline_stage = world_size - 1

    def construct(self, input_ids, labels=None, attention_mask=None):
        """ pipeline test net forward """
        output_dict = self.preprocess(input_ids, labels, attention_mask)
        input_ids = output_dict["input_ids"]
        attention_mask = output_dict["attention_mask"]
        labels = output_dict["labels"]
        hidden_stage, embedding_table = self.embedding(input_ids)
        for i in range(len(self.fake_transformer_layers)):
            hidden_stage = self.fake_transformer_layers[i](hidden_stage, attention_mask)
        hidden_stage = self.final_norm(hidden_stage)
        if not self.share_embedding_weight:
            embedding_table = None
        logits = self.fake_head(hidden_stage, embedding_table)
        loss = self.total_loss(logits, labels)
        return loss
