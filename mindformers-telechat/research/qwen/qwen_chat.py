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
"""Helper functions to make 'chat' feature work on Mindformers' Qwen port."""

from typing import Tuple, List, Union, Optional

from mindspore.common.tensor import Tensor

HistoryType = List[Tuple[str, str]]
TokensType = List[int]


def get_stop_words_ids(chat_format, tokenizer):
    """get ids of stop words"""
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
        tokenizer,
        query: str,
        history: HistoryType = None,
        system: str = '',
        max_window_size: int = 6144,
        chat_format: str = 'chatml',
        verbose: bool = False
) -> Tuple[str, TokensType]:
    """make chat context"""
    if not history:
        history = []

    if chat_format == 'raw':
        prompt_text = query
        prompt_tokens = tokenizer.encode(prompt_text)
        return prompt_text, prompt_tokens

    if chat_format == 'chatml':
        # prompt example
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # <|im_start|>user\n你好<|im_end|>\n
        # <|im_start|>assistant\n你好！很高兴为你提供帮助。<|im_end|>\n
        # <|im_start|>user\n给我讲一个年轻人奋斗创业最终取得成功的故事。<|im_end|>\n
        # <|im_start|>assistant\n

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            tokens = tokenizer.encode(role, allowed_special=set()) \
                     + nl_tokens \
                     + tokenizer.encode(content, allowed_special=set())
            return f"{role}\n{content}", tokens

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        history_text = ''
        history_tokens = []

        # add history chats
        # reverse(history): make sure latest dialogs added in case 'max_window_size' exceeded
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            turn_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            turn_text = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            total_size = len(system_tokens) + len(turn_tokens) + len(history_tokens)
            if total_size < max_window_size:
                history_tokens = turn_tokens + history_tokens
                history_text = turn_text + history_text
            else:
                break

        prompt_tokens = system_tokens + history_tokens + (nl_tokens + im_start_tokens + _tokenize_str("user", query)[
            1] + im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens)

        prompt_text = f"{im_start}{system_text}{im_end}" + history_text + \
                      f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        if verbose:
            print("\nInput: ", prompt_text)

        return prompt_text, prompt_tokens
    raise NotImplementedError(f"Unknown chat format {chat_format!r}")


def _decode_default(
        tokens: List[int],
        *,
        stop_words: List[str],
        eod_words: List[str],
        tokenizer,
        raw_text_len: int,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = 'replace',
):
    """decode default"""
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate: ", trim_decode_tokens)

    end_reason = f"Gen length {len(tokens)}"
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f"Gen {eod_word!r}"
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nEnd Reason:", end_reason)
        print("\nGenerate: ", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    return trim_decode_tokens


def _decode_chatml(
        tokens: List[int],
        *,
        stop_words: List[str],
        eod_token_ids: List[int],
        tokenizer,
        raw_text_len: int,
        context_length: int,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = 'replace'
):
    """decode chatml"""
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    return trim_decode_tokens


def decode_tokens(
        tokens: Union[Tensor, TokensType],
        tokenizer,
        raw_text_len: int,
        context_length: int,
        chat_format: str,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = "replace"
) -> str:
    """decode tokens"""
    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    if chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=["<|endoftext|>"],
            eod_words=["<|endoftext|>"],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    raise NotImplementedError(f"Unknown chat format {chat_format!r}")


def chat(
        model,
        tokenizer,
        query: str,
        history: Optional[List[Tuple[str, str]]],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stop_words_ids: Optional[List[TokensType]] = None,
        **kwargs,
) -> Tuple[str, HistoryType]:
    """do chat"""
    if not history:
        history = []
    if not stop_words_ids:
        stop_words_ids = []

    max_new_tokens = kwargs.get('max_new_tokens', model.transformer.seq_length // 4)
    max_window_size = kwargs.get('max_window_size',
                                 model.transformer.seq_length - max_new_tokens - 48)
    chat_format = kwargs.get('chat_format', 'chatml')
    verbose = kwargs.get('verbose', False)

    prompt_text, prompt_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system=system,
        max_window_size=max_window_size,
        chat_format=chat_format,
        verbose=verbose
    )
    stop_words_ids.extend(get_stop_words_ids(chat_format, tokenizer))

    outputs = model.generate(
        [prompt_tokens],
        max_new_tokens=max_new_tokens
    )

    response = decode_tokens(
        outputs[0],
        tokenizer,
        raw_text_len=len(prompt_text),
        context_length=len(prompt_tokens),
        chat_format=chat_format,
        verbose=verbose,
        errors='replace'
    )

    if append_history:
        history.append((query, response))

    return response, history
