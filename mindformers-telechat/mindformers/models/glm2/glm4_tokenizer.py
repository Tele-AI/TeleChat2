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
"""ChatGLM4 Tokenizer."""
import base64
import os
import json
from typing import List, Optional, Union, Dict
import tiktoken
import regex as re
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import PreTrainedTokenizer, PaddingStrategy, EncodedInput, BatchEncoding
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister

__all__ = ['ChatGLM4Tokenizer']


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class ChatGLM4Tokenizer(PreTrainedTokenizer):
    """
    Construct a ChatGLM4 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (str): The vocabulary file path.
        clean_up_tokenization_spaces (bool, optional): Whether to delete redundant spaces. Default: ``False``.
        encode_special_tokens (bool, optional): Whether to encode the special tokens. Default: ``False``.
        eos_token (Union[str, tokenizers.AddedToken], optional): The end of sequence token. Default: `"<|endoftext|>"` .
        pad_token (Union[str, tokenizers.AddedToken], optional): A special token used to make arrays of tokens the same
            size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
            Default: `"<|endoftext|>"` .
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Returns:
        A ChatGLM4Tokenizer instance.

    Examples:
        >>> from mindformers import ChatGLM4Tokenizer
        >>> tokenizer = ChatGLM4Tokenizer('tokenizer.model')
        >>> prompts = ["晚上睡不着应该怎么办"]
        >>> token_id = tokenizer(prompts)
        >>> input_ids = token_id['input_ids']
        >>> print(input_ids)
        [[151331, 151333, 101160, 120410, 99379, 103298]]
        >>> response = tokenizer.decode(input_ids)
        >>> print(response)
        ['晚上睡不着应该怎么办']
    """
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    _support_list = MindFormerBook.get_tokenizer_support_list()['glm4']

    def __init__(
            self,
            vocab_file,
            clean_up_tokenization_spaces=False,
            encode_special_tokens=False,
            eos_token='<|endoftext|>',
            pad_token='<|endoftext|>',
            **kwargs
    ):
        self.name = "GLM4Tokenizer"
        self.vocab_file = vocab_file
        pat_str = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
                   "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
        self.pat_str = re.compile(pat_str)
        self.encode_special_tokens = encode_special_tokens

        mergeable_ranks = {}
        self.special_tokens = {"<|endoftext|>": 151329, "[MASK]": 151330, "[gMASK]": 151331, "[sMASK]": 151332,
                               "<sop>": 151333, "<eop>": 151334, "<|system|>": 151335, "<|user|>": 151336,
                               "<|assistant|>": 151337, "<|observation|>": 151338, "<|begin_of_image|>": 151339,
                               "<|end_of_image|>": 151340, "<|begin_of_video|>": 151341, "<|end_of_video|>": 151342}

        self._eos_token = eos_token
        self._pad_token = pad_token

        with open(vocab_file) as f:
            for line in f:
                token, rank = line.strip().split()
                rank = int(rank)
                token = base64.b64decode(token)
                mergeable_ranks[token] = rank

        self.mergeable_ranks = mergeable_ranks

        self.tokenizer = tiktoken.Encoding(
            name="my_tokenizer",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={}
        )

        self.decoder = {rank: token for token, rank in mergeable_ranks.items()}
        self.n_words = len(self.decoder)

        super().__init__(
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )
        for token in self.special_tokens:
            self.add_tokens(token, special_tokens=True)

    @property
    def vocab_size(self):
        return self.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str, int]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, int):
                t = chr(t)
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors="replace")
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type int, bytes or str")
        if temp != "":
            text += temp.decode("utf-8", errors="replace")
        return text

    def _tokenize(self, text, **kwargs):
        tokens = []
        ids = self.tokenizer.encode(text)
        for t in ids:
            tokens.append(self.decoder[t])
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.mergeable_ranks.get(token)

    def convert_special_tokens_to_ids(self, token):
        """ Converts special tokens to ids using the vocab. """
        try:
            return self.special_tokens.get(token)
        except ValueError as e:
            raise ValueError(f"{token} is not a special token for {self.name}") from e

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, "")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
        Returns:
            `Tuple(str)`, Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, self.vocab_files_names.get("vocab_file"))
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(vocab_file, flags_, 0o750), 'wb') as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.convert_tokens_to_ids("[gMASK]"), self.convert_tokens_to_ids("<sop>")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message, tokenize=True):
        """build single message with role."""
        if role not in ["system", "user", "assistant", "observation"]:
            raise ValueError(f'{role} not in ["system", "user", "assistant", "observation"]')
        if tokenize:
            role_tokens = [self.convert_tokens_to_ids(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n",
                                                                                              disallowed_special=())
            message_tokens = self.tokenizer.encode(message, disallowed_special=())
            tokens = role_tokens + message_tokens
            return tokens
        return str(f"<|{role}|>{metadata}\n{message}")

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`, list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.convert_tokens_to_ids("<eos>")]
        return token_ids_0

    def build_batch_input(self, queries, histories=None, roles="user", padding=True, return_tensors="np"):
        """build batch input with role."""
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            raise TypeError(f'{queries} must be of type list!')
        batch_size = len(queries)
        if isinstance(roles, str):
            roles = [roles] * batch_size
        if isinstance(histories, list) and len(histories) != batch_size:
            histories = [histories]
        if histories is None:
            histories = [[] for _ in range(batch_size)]

        if batch_size != len(histories) or batch_size != len(roles):
            raise ValueError(f'len(queries) should equals to len(roles) and len(histories), but got len(queries): '
                             f'{len(queries)} and len(histories):{len(histories)} and len(roles): {len(roles)}')
        batch_inputs = []
        for query, history, role in zip(queries, histories, roles):
            if history is None:
                history = []
            input_ids = []
            for item in history:
                content = item["content"]
                if item["role"] == "system" and "tools" in item:
                    content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
                input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
            input_ids.extend(self.build_single_message(role, "", query))
            input_ids.extend([self.convert_special_tokens_to_ids("<|assistant|>")])
            batch_inputs.append(input_ids)

        return self.batch_encode_plus(batch_inputs, return_tensors=return_tensors,
                                      is_split_into_words=True, padding=padding)

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = encoded_inputs["position_ids"] + [0] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference

        return encoded_inputs

    def build_chat_input(self, query, history=None, role="user", return_tensors="np"):
        """build chat input with role."""
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.convert_special_tokens_to_ids("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors=return_tensors, is_split_into_words=True)

    # pylint: disable=W0221
    def apply_chat_template(self, conversation, return_tensors=None, **tokenizer_kwargs):
        if not conversation:
            return []
        if not (isinstance(conversation, list) and len(conversation) == 1
                and isinstance(conversation[0], Dict)):
            raise ValueError(f"conversation {conversation} is invalid.")
        return self.build_chat_input(query=conversation[0].get("content"), role=conversation[0].get("role"),
                                     return_tensors=return_tensors)["input_ids"][0]
