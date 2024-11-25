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
""" test tokenizer """
import os
import shutil

import pytest
from mindformers import AutoTokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_level_zero_tokenizers():
    """
    Feature: Test all tokenizers.
    Description: Test all tokenizers, including save_pretrained, form_pretrained, tokenizer and return_tensors
    Expectation: No exception
    """
    tokenizer_list = ["gpt2", "glm2_6b", "llama2_7b"]
    string_list = ["I love Beijing.", "我爱北京。"]
    return_tensors_sig = ["", "ms", "np"]

    all_res = [
        [40, 1842, 11618, 13],  # gpt2 english
        [22755, 239, 163, 230, 109, 44293, 245, 12859, 105, 16764],  # gpt2 chinese
        [307, 1379, 13924, 30930],  # glm2 english
        [34211, 54799, 31719, 31155],  # glm2 chinese
        [306, 5360, 1522, 823, 292, 29889],  # llama2 english
        [29871, 30672, 234, 139, 180, 30662, 30675, 30267]  # llama2 chinese
    ]

    check_tokenizers(tokenizer_list, string_list, return_tensors_sig, all_res)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_level_zero_special_tokens():
    """
    Feature: Test special_tokens.
    Description: Test all special_tokens and special_ids cache after add_tokens.
    Expectation: No exception.
    """

    tokenizer_list = ["gpt2", "glm2_6b", "llama2_7b"]
    special_tokens_list = ["<s>", "</s>", "<unk>", "<pad>", "<cls>",
                           "<extra_id_77>", "<extra_id_88>", "<extra_id_99>"]

    check_special_tokens(tokenizer_list, special_tokens_list)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_level_one_tokenizers():
    """
    Feature: Test all tokenizers.
    Description: Test all tokenizers, including save_pretrained, form_pretrained, tokenizer and return_tensors
    Expectation: No exception
    """
    tokenizer_list = ["bert_base_uncased", "llama_7b", "bloom_560m",
                      "pangualpha_2_6b", "clip_vit_b_32", "glm_6b", "t5_small"]
    string_list = ["I love Beijing.", "我爱北京。"]
    return_tensors_sig = ["", "ms", "np"]

    all_res = [
        [1045, 2293, 7211, 1012],  # bert english
        [1855, 100, 1781, 1755, 1636],  # bert chinese
        [312, 1545, 11285, 31843],  # llama english
        [31822, 233, 139, 148, 234, 139, 180, 232, 143, 154, 231, 189, 175, 230, 131, 133],  # llama chinese
        [44, 19134, 61335, 17],  # bloom english
        [164830, 11385, 420],  # bloom chinese
        [2787, 13, 4, 13, 28420, 13, 4, 37472, 34],  # pangualpha english
        [21, 453, 263, 12],  # pangualpha chinese
        [328, 793, 11796, 269],  # clip english
        [162, 230, 239, 163, 230, 109, 161, 234, 245, 21078, 361, 38000],  # clip chinese
        [115, 703, 8994, 7],  # glm english
        [5, 76202, 64241, 63823],  # glm chinese
        [27, 333, 14465, 5],  # t5 english
        [3, 2],  # t5 chinese
    ]

    check_tokenizers(tokenizer_list, string_list, return_tensors_sig, all_res)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_level_one_special_tokens():
    """
    Feature: Test special_tokens.
    Description: Test all special_tokens and special_ids cache after add_tokens.
    Expectation: No exception.
    """

    tokenizer_list = ["bert_base_uncased", "llama_7b", "bloom_560m",
                      "pangualpha_2_6b", "clip_vit_b_32", "glm_6b", "t5_small"]
    special_tokens_list = ["<s>", "</s>", "<unk>", "<pad>", "<cls>",
                           "<extra_id_77>", "<extra_id_88>", "<extra_id_99>"]

    check_special_tokens(tokenizer_list, special_tokens_list)


def get_tokenizer(tokenizer_item):
    """get tokenizer with retry"""
    retry = True
    tokenizer = None
    count = 0
    while retry:
        try:
            count += 1
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_item)
            retry = False
        # pylint: disable=W0703
        except Exception as e:
            if os.path.exists(f"./checkpoint_download/{tokenizer_item}"):
                shutil.rmtree(f"./checkpoint_download/{tokenizer_item}")
            print(f"{tokenizer_item} tokenizer initialize failed, due to \"{e}\".")
        if count >= 3:
            retry = False
    if tokenizer is None:
        raise RuntimeError(f"{tokenizer_item} tokenizer initialize failed for {count} times.")

    return tokenizer


def check_tokenizers(tokenizer_list, string_list, return_tensors_sig, all_res):
    """test_tokenizers"""
    for i, tokenizer_item in enumerate(tokenizer_list):
        tokenizer = get_tokenizer(tokenizer_item)
        for j, string in enumerate(string_list):
            for return_tensor_sig in return_tensors_sig:
                if not return_tensor_sig:
                    result = tokenizer(string, add_special_tokens=False)
                    print("{}: the result of {} is {}".format(tokenizer_item, string, result))
                    assert result.input_ids == all_res[2 * i + j]
                    result = tokenizer(string, padding="max_length", max_length=100, add_special_tokens=False)
                    assert result.input_ids[:len(all_res[2 * i + j])] == all_res[2 * i + j]
                    print("{}: the pad result of {} is {}".format(tokenizer_item, string, result))
                else:
                    result = tokenizer(string, return_tensors=return_tensor_sig, add_special_tokens=False)
                    if return_tensor_sig == "ms":
                        assert result.input_ids.asnumpy().tolist() == all_res[2 * i + j]
                    else:
                        assert result.input_ids.tolist() == all_res[2 * i + j]
                    print("{}: the {} result of {} is {}".format(tokenizer_item, return_tensor_sig, string, result))
                    result = tokenizer(string, padding="max_length", max_length=100, return_tensors=return_tensor_sig,
                                       add_special_tokens=False)
                    if return_tensor_sig == "ms":
                        assert result.input_ids.asnumpy().tolist()[:len(all_res[2 * i + j])] == all_res[2 * i + j]
                    else:
                        assert result.input_ids.tolist()[:len(all_res[2 * i + j])] == all_res[2 * i + j]
                    print("{}: the {} pad result of {} is {}".format(tokenizer_item, return_tensor_sig, string, result))


def check_special_tokens(tokenizer_list, special_tokens_list):
    """test special tokens"""
    for _, tokenizer_item in enumerate(tokenizer_list):
        tokenizer = get_tokenizer(tokenizer_item)
        default_all_special_tokens = tokenizer.all_special_tokens
        default_all_special_ids = tokenizer.all_special_ids
        cached_all_special_tokens = tokenizer.all_special_tokens
        cached_all_special_ids = tokenizer.all_special_ids
        assert default_all_special_tokens == cached_all_special_tokens
        assert default_all_special_ids == cached_all_special_ids

        print("{}: cached_special_tokens is {}, cached_special_ids is {}"
              .format(tokenizer_item, cached_all_special_tokens, cached_all_special_ids))

        tokenizer.add_tokens(special_tokens_list, special_tokens=True)

        added_all_special_tokens = tokenizer.all_special_tokens
        added_all_special_ids = tokenizer.all_special_ids
        assert set(default_all_special_tokens) < set(added_all_special_tokens)
        assert set(default_all_special_ids) < set(added_all_special_ids)

        cached_all_special_tokens = tokenizer.all_special_tokens
        cached_all_special_ids = tokenizer.all_special_ids
        assert added_all_special_tokens == cached_all_special_tokens
        assert added_all_special_ids == cached_all_special_ids

        print("{}: after add_tokens {}, cached_special_tokens is {}, cached_special_ids is {}"
              .format(tokenizer_item, special_tokens_list, cached_all_special_tokens, cached_all_special_ids))
