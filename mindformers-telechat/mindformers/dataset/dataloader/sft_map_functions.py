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
"""Map functions for the SFT data."""
import numpy as np


def _prepare_for_model(tokenizer, max_length, prompt, answer=None):
    """Prepare input data for model fine-tuning or evaluation."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    pair_ids = tokenizer.encode(answer, add_special_tokens=False) if answer else None
    return tokenizer.prepare_for_model(ids=ids,
                                       pair_ids=pair_ids,
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True,
                                       truncate_direction="LEFT",
                                       return_attention_mask=True)


def default_map_fn(example, **kwargs):
    """Default data parsing function."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    values = list(example.values())
    if len(values) == 1:
        result = _prepare_for_model(tokenizer, max_length, values[0])
    else:
        result = _prepare_for_model(tokenizer, max_length, values[0], values[1])
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"])


def alpaca_map_fn(example, **kwargs):
    """Parsing the Alpaca dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    if example.get("input"):
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ).format_map(example)
    else:
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ).format_map(example)
    result = _prepare_for_model(tokenizer, max_length, text, example.get("output"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"])


def advertisegen_map_fn(example, **kwargs):
    """Parsing the AdvertiseGen dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    result = _prepare_for_model(tokenizer, max_length, example.get("content"), example.get("summary"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"])


def cola_map_fn(example, **kwargs):
    """Parsing the COLA dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    values = list(example.values())
    result = _prepare_for_model(tokenizer, max_length, values[3])
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"], labels=values[1])


def imdb_map_fn(example, **kwargs):
    """Parsing the IMDB dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    label = 1 if example.get("sentiment") == 'positive' else 0
    result = _prepare_for_model(tokenizer, max_length, example.get("review"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"], labels=label)


def sst2_map_fn(example, **kwargs):
    """Parsing the SST-2 dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    result = _prepare_for_model(tokenizer, max_length, example.get("sentence"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"], labels=example.get("label"))


def agnwes_map_fn(example, **kwargs):
    """Parsing the AG-News dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    result = _prepare_for_model(tokenizer, max_length, example.get("sentence"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"], labels=example.get("label"))


def tnews_map_fn(example, **kwargs):
    """Parsing the TNEWS dataset."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    label = int(example.get("label")) - 100
    result = _prepare_for_model(tokenizer, max_length, example.get("sentence"))
    return dict(input_ids=result["input_ids"], attention_mask=result["attention_mask"], labels=label)


def multi_round_chat_map_fn(example, **kwargs):
    """Parsing the dataset of multiple rounds of chat."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    data_field = kwargs.get("data_field", "conversations")
    from_keyword, value_keyword = kwargs.get("from_keyword", "from"), kwargs.get("value_keyword", "value")
    user_role_name = kwargs.get("user_role_name", "human")
    assistant_role_name = kwargs.get("assistant_role_name", "gpt")
    user_prompt, assistant_prompt = kwargs.get("user_prompt", ""), kwargs.get("assistant_prompt", "")
    ignore_token_id = kwargs.get("ignore_token_id", -100)

    raw_input_id = []
    raw_label = []
    for message in example[data_field]:
        from_ = message[from_keyword]
        value = message[value_keyword]
        if from_ == user_role_name:
            value_ids = tokenizer.encode(user_prompt + value, add_special_tokens=False)
            raw_input_id += value_ids
            raw_label += [ignore_token_id] * len(value_ids)
        elif from_ == assistant_role_name:
            value_ids = tokenizer.encode(assistant_prompt + value, add_special_tokens=False)
            raw_input_id += value_ids
            raw_label += value_ids
        else:
            raise ValueError(f"Incorrect role name: {from_}. Check the values of `user_role_name` "
                             f"and `assistant_role_name` in `map_function_kwargs`.")

    raw_input_id.append(tokenizer.eos_token_id)
    raw_label.append(tokenizer.eos_token_id)

    if len(raw_input_id) >= max_length:
        input_id = raw_input_id[: max_length]
        attention_mask = [1] * max_length
        label = raw_label[: max_length]
    else:
        input_id = raw_input_id + [tokenizer.pad_token_id] * (max_length - len(raw_input_id))
        attention_mask = [1] * len(raw_input_id) + [0] * (max_length - len(raw_input_id))
        label = raw_label + [ignore_token_id] * (max_length - len(raw_label))
    return dict(input_ids=input_id, attention_mask=attention_mask, labels=label)


def multi_round_chat_dyn_map_fn(example, **kwargs):
    """Parsing the dataset of multiple rounds of chat."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    data_field = kwargs.get("data_field", "conversations")
    from_keyword, value_keyword = kwargs.get("from_keyword", "from"), kwargs.get("value_keyword", "value")
    user_role_name = kwargs.get("user_role_name", "human")
    assistant_role_name = kwargs.get("assistant_role_name", "gpt")
    user_prompt, assistant_prompt = kwargs.get("user_prompt", ""), kwargs.get("assistant_prompt", "")
    user_prompt_role = kwargs.get("user_prompt_role", "")
    assistant_prompt_role = kwargs.get("assistant_prompt_role", "")
    ignore_token_id = kwargs.get("ignore_token_id", -100)
    bos_token, sep_token = kwargs.get("bos_token", "<s>"), kwargs.get("sep_token", "")

    raw_input_id = []
    raw_label = []
    for message in example[data_field]:
        from_ = message[from_keyword]
        value = message[value_keyword]
        if from_ == user_role_name:
            user_value_ids = tokenizer.encode(bos_token + user_prompt + user_prompt_role + value,
                                              add_special_tokens=False)
            assistant_value_ids = tokenizer.encode(assistant_prompt + assistant_prompt_role, add_special_tokens=False)
            value_ids = user_value_ids + assistant_value_ids
            raw_input_id += value_ids
            raw_label += [ignore_token_id] * len(value_ids)
        elif from_ == assistant_role_name:
            value = sep_token + value
            value_ids = tokenizer.encode(value, add_special_tokens=False)
            raw_input_id += value_ids
            raw_label += value_ids
        else:
            raise ValueError(f"Incorrect role name: {from_}. Check the values of `user_role_name` "
                             f"and `assistant_role_name` in `map_function_kwargs`.")
    raw_input_id.append(tokenizer.eos_token_id)
    raw_label.append(tokenizer.eos_token_id)

    if len(raw_input_id) >= max_length:
        input_id = raw_input_id[: max_length]
        attention_mask = [1] * max_length
        label = raw_label[: max_length]
    else:
        input_id = raw_input_id
        attention_mask = [1] * len(raw_input_id)
        label = raw_label + [ignore_token_id] * (len(raw_input_id) - len(raw_label))
    return dict(input_ids=input_id, attention_mask=attention_mask, labels=label)


def multi_instruct_dyn_map_fn(example, **kwargs):
    """Parsing the dataset of init multiple rounds of chat."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    ignore_token_id = kwargs.get("ignore_token_id", -100)
    from_keyword, value_keyword = kwargs.get("from_keyword", "input"), kwargs.get("value_keyword", "output")
    instruction_keyword = kwargs.get("instruction_keyword", "instruction")
    user_prompt_role = kwargs.get("user_prompt_role", "")
    assistant_prompt_role = kwargs.get("assistant_prompt_role", "")

    raw_input_id = []
    raw_label = []
    instruction = example[instruction_keyword]
    input_ = example[from_keyword]
    output = example[value_keyword]

    instruction_ids = tokenizer.encode(user_prompt_role + instruction + input_ + assistant_prompt_role,
                                       add_special_tokens=False)
    raw_input_id += instruction_ids
    raw_label += [ignore_token_id] * len(instruction_ids)

    value_ids = tokenizer.encode(output, add_special_tokens=False)
    raw_input_id += value_ids
    raw_label += value_ids

    raw_input_id.append(tokenizer.eos_token_id)
    raw_label.append(tokenizer.eos_token_id)

    if len(raw_input_id) >= max_length:
        input_id = raw_input_id[: max_length]
        attention_mask = [1] * max_length
        label = raw_label[: max_length]
    else:
        input_id = raw_input_id
        attention_mask = [1] * len(raw_input_id)
        label = raw_label + [ignore_token_id] * (len(raw_input_id) - len(raw_label))
    return dict(input_ids=input_id, attention_mask=attention_mask, labels=label)


def multi_round_chat_dyn_map_fn_alpaca(example, **kwargs):
    """Parsing the dataset of multiple rounds of chat."""
    tokenizer, max_length = kwargs.get("tokenizer"), kwargs.get("max_length")
    data_field = kwargs.get("data_field", "conversations")
    from_keyword, value_keyword = kwargs.get("from_keyword", "from"), kwargs.get("value_keyword", "value")
    user_role_name = kwargs.get("user_role_name", "human")
    assistant_role_name = kwargs.get("assistant_role_name", "gpt")
    user_prompt, assistant_prompt = kwargs.get("user_prompt", ""), kwargs.get("assistant_prompt", "")
    user_prompt_role, assistant_prompt_role = kwargs.get("user_prompt_role", ""), \
    kwargs.get("assistant_prompt_role", "")
    bos_token, sep_token = kwargs.get("bos_token", "<|im_start|>"), kwargs.get("sep_token", "<|im_end|>\n<|im_start|>")

    raw_input_id = []
    raw_label = []
    for message in example[data_field]:
        from_ = message[from_keyword]
        value = message[value_keyword]
        if from_ == user_role_name:
            user_value_ids = tokenizer.encode(bos_token + user_prompt + sep_token + user_prompt_role + \
            value + sep_token, add_special_tokens=False)
            assistant_value_ids = tokenizer.encode(assistant_prompt + assistant_prompt_role, add_special_tokens=False)
            value_ids = user_value_ids + assistant_value_ids
            raw_input_id += value_ids
            raw_label += value_ids
        elif from_ == assistant_role_name:
            value = value + "<|im_end|>\n"
            value_ids = tokenizer.encode(value, add_special_tokens=False)
            raw_input_id += value_ids
            raw_label += value_ids
        else:
            raise ValueError(f"Incorrect role name: {from_}. Check the values of `user_role_name` "
                             f"and `assistant_role_name` in `map_function_kwargs`.")
    raw_input_id.append(tokenizer.eos_token_id)
    raw_label.append(tokenizer.eos_token_id)

    if len(raw_input_id) >= max_length:
        input_id = raw_input_id[: max_length]
        attention_mask = [1] * max_length
        label = raw_label[: max_length]
    else:
        input_id = raw_input_id
        attention_mask = [1] * len(raw_input_id)
        label = raw_label
    attention_mask = np.where(input_id == tokenizer.pad_token_id, 0, 1).astype(np.int32)
    return dict(input_ids=input_id, attention_mask=attention_mask, labels=label)

_SFT_MAP_FUNCTIONS = {
    "default": default_map_fn,
    "alpaca": alpaca_map_fn,
    "advertisegen": advertisegen_map_fn,
    "cola": cola_map_fn,
    "imdb": imdb_map_fn,
    "sst-2": sst2_map_fn,
    "ag-news": agnwes_map_fn,
    "tnews": tnews_map_fn,
    "multi-round-chat": multi_round_chat_map_fn,
    "multi-round-chat-dyn": multi_round_chat_dyn_map_fn,
    "multi-round-chat-dyn-alpaca": multi_round_chat_dyn_map_fn_alpaca,
    "multi-instruct-dyn": multi_instruct_dyn_map_fn,
}
