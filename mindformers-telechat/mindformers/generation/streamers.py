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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
# ============================================================================
"""Streamers for text generation."""
from multiprocessing import Queue
from typing import Optional
import numpy as np
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ['BaseStreamer', 'TextStreamer', 'TextIteratorStreamer']


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        >>> from mindformers import GPT2LMHeadModel, GPT2Tokenizer, TextStreamer

        >>> tok = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
        An increasing sequence: one, two, three, four, five, six, seven, eight,
    """

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 skip_prompt: bool = False,
                 skip_special_tokens: bool = True,
                 **decode_kwargs):
        Validator.check_value_type("skip_prompt", skip_prompt, [bool], self.__class__.__name__)
        Validator.check_value_type("skip_special_tokens", skip_special_tokens, [bool], self.__class__.__name__)
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.batch_stream = False
        self.token_cache = []
        self.text_cache = ""
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, int):
            self.token_cache.append(value)
        elif isinstance(value, list):
            if len(value) > 1 and isinstance(value[0], list):
                # switch from single mode to batch mode
                if not self.batch_stream:
                    self.batch_stream = True
                    self.text_cache = ""
            else:
                # batch that equals 1
                if len(value) == 1 and isinstance(value[0], list):
                    value = value[0]
                # switch from batch mode to single mode
                if self.batch_stream:
                    self.batch_stream = False
                    self.token_cache = []
                    self.print_len = 0
                self.token_cache.extend(value)
        else:
            raise ValueError(f"TextStreamer only supports int, or 1 ~ 2 dim numpy.ndarray/list as inputs, "
                             f"but got {type(value)} with value {value}.")

        # Add the new token to the cache and decodes the entire thing.
        if self.batch_stream:
            text = self.tokenizer.batch_decode(value, self.skip_special_tokens, **self.decode_kwargs)
        else:
            text = self.tokenizer.decode(self.token_cache, self.skip_special_tokens, **self.decode_kwargs)

        printable_text = self.get_printable_text(text)
        self.on_finalized_text(printable_text)

    def get_printable_text(self, text):
        """Get printable text when a new element comes in."""
        # for batch streamer, we directly return the text
        if self.batch_stream:
            return text
        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif text and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        return printable_text

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if not self.batch_stream and self.token_cache:
            text = self.tokenizer.decode(self.token_cache, self.skip_special_tokens, **self.decode_kwargs)
            printable_text = text[self.print_len :]
        else:
            printable_text = ""

        self.on_finalized_text(printable_text, stream_end=True)

        # always reset values when stream ends
        self.next_tokens_are_prompt = True
        self.token_cache = []
        self.text_cache = ""
        self.print_len = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if self.batch_stream:
            if not self.text_cache:
                self.text_cache = text
            elif text:
                self.text_cache = [i + j for i, j in zip(self.text_cache, text)]
            print(f'\r{self.text_cache}', flush=True, end="" if not stream_end else None)
        else:
            print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all the other languages.
        if (
                (0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)
        ):
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        >>> from mindformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(input_ids=inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        An increasing sequence: one, two, three, four, five, six, seven, eight,
    """

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 skip_prompt: bool = False,
                 timeout: Optional[float] = None,
                 **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if text:
            self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def clear(self):
        while not self.text_queue.empty():
            self.text_queue.get()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            return None
        return value
