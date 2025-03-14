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
Test llama parallel decoding forward.
How to run this:
    pytest tests/st/test_model/test_llama_parallel_decoding_model/test_llama_parallel_decoding_forward.py
"""
import os
import numpy as np
import mindspore as ms

from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


class TestLlamaParallelDecodingForward:
    """A test class for testing parallel decoding forward."""

    def test_forward(self):
        """
        Feature: Parallel decoding
        Description: Test llama parallel decoding forward.
        Expectation: AssertionError
        """
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "PagedAttention,FlashAttentionScore"
        os.environ["RUN_MODE"] = "predict"
        os.environ["ASCEND_HOME_PATH"] = "/usr/local/Ascend/latest"
        ms.set_seed(0)
        model_config = LlamaConfig(
            num_layers=1,
            hidden_size=4,
            num_heads=2,
            seq_length=512,
            use_past=True,
            vocab_size=2,
            num_blocks=16,
            use_flash_attention=True,
            is_dynamic=True,
            parallel_decoding_params={"parallel_decoding": "la"},
        )
        model = LlamaForCausalLM(model_config)
        model.set_dynamic_inputs()

        # prefill
        bs = 4
        seq_len = 16
        input_ids = np.random.randint(0, 2, size=(bs, seq_len), dtype=np.int32)
        valid_length_each_example = np.array([seq_len] * bs, np.int32)
        block_tables = np.arange(0, stop=int(model_config.num_blocks // bs * bs), dtype=np.int32).reshape(bs, -1)
        slot_mapping = np.arange(0, stop=seq_len * bs, dtype=np.int32)
        position_ids = None
        spec_mask = None
        q_seq_lens = None
        res, _ = model.forward(
            input_ids=input_ids,
            valid_length_each_example=valid_length_each_example,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            prefill=True,
            use_past=True,
            position_ids=position_ids,
            spec_mask=spec_mask,
            q_seq_lens=q_seq_lens,
        )
        expect = np.array([
            [-0.01670837, 0.03088379],
            [0.021636966, -0.03994751],
            [-0.01670837, 0.03088379],
            [0.021148682, -0.03915405],
        ], np.float16)
        assert np.allclose(res.numpy(), expect, atol=1e-3)
        assert res.shape == (bs, model_config.vocab_size)

        # increment
        model.add_flags_custom(is_first_iteration=False)
        expect_list = [
            np.array([
                [0.02194214, -0.03997803],
                [0.02194214, -0.03997803],
            ], np.float16),
            np.array([
                [0.02194214, -0.03997803],
                [-0.0163269, 0.03062439],
                [0.02194214, -0.03997803],
            ], np.float16),
            np.array([
                [-0.0163269, 0.03062439],
                [-0.0163269, 0.03062439],
            ], np.float16),
        ]
        for expect in expect_list:
            bs = np.random.randint(1, 3)
            seq_len = np.random.randint(1, 4)
            token_num = bs * seq_len
            batch_valid_length = np.random.randint(token_num, 64)
            input_ids = np.random.randint(0, 2, size=(token_num, 1), dtype=np.int32)
            valid_length_each_example = np.array([batch_valid_length] * bs, np.int32)
            block_tables = np.arange(0, stop=int(model_config.num_blocks // bs * bs), dtype=np.int32).reshape(bs, -1)
            slot_mapping = np.arange(0, stop=token_num, dtype=np.int32)
            position_ids = np.random.randint(0, 2, size=(token_num,), dtype=np.int32)
            spec_mask = np.random.randint(0, 2, size=(token_num, batch_valid_length)).astype(np.float16)
            q_seq_lens = np.array([seq_len] * bs, np.int32)
            res, _ = model.forward(
                input_ids=input_ids,
                valid_length_each_example=valid_length_each_example,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                prefill=False,
                use_past=True,
                position_ids=position_ids,
                spec_mask=spec_mask,
                q_seq_lens=q_seq_lens,
            )
            assert np.allclose(res.numpy(), expect, atol=1e-3)
            assert res.shape == (token_num, model_config.vocab_size)
