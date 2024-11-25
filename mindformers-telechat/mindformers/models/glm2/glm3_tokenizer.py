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
"""ChatGLM3 Tokenizer."""
import os
import json
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import PreTrainedTokenizer, PaddingStrategy, EncodedInput, BatchEncoding
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister

__all__ = ['ChatGLM3Tokenizer']


class SPTokenizer:
    """Tokenizer process special tokens."""

    def __init__(self, model_path: str):
        # reload tokenizer
        if not os.path.isfile(model_path):
            raise ValueError(f"{model_path} is not file.")
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        if self.sp_model.vocab_size() != self.sp_model.get_piece_size():
            raise ValueError(f"self.sp_model.vocab_size() should equal to self.sp_model.get_piece_size(), "
                             f"but sp_model.vocab_size() got {self.sp_model.vocab_size()}, "
                             f"sp_model.get_piece_size() got {self.sp_model.get_piece_size()}.")

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop", "<|system|>", "<|user|>", "<|assistant|>",
                          "<|observation|>"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    # pylint: disable=W0613
    def tokenize(self, s: str, pair=None, add_special_tokens=True, **kwargs):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        if not isinstance(s, str):
            raise ValueError(f"s should be str, but got {s}.")
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    # pylint: disable=W0613
    def decode(self, t: List[int], skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs) -> str:
        """Convert the token into a readable text string"""
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class ChatGLM3Tokenizer(PreTrainedTokenizer):
    """
    Construct a ChatGLM3 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file(str): The vocabulary file path.
        bos_token (str, tokenizers.AddedToken): The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token. Default: `"<sop>"` .
        eos_token (str, tokenizers.AddedToken): The end of sequence token. Default: `"</s>"` .
        end_token (str, tokenizers.AddedToken): The end of sequence token. Default: `"</s>"` .
        mask_token (str, tokenizers.AddedToken): The masked token. Default: `"[MASK]"` .
        gmask_token (str, tokenizers.AddedToken): The special masked token. Default: ``"[gMASK]"`` .
        pad_token (str, tokenizers.AddedToken): A special token used to make arrays of tokens the same size for batching
            purpose. Will then be ignored by attention mechanisms or loss computation. Default: `"<pad>"` .
        unk_token (str, tokenizers.AddedToken): The unknown token. Default: `"<unk>"` .
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Returns:
        A ChatGLM3Tokenizer instance.

    Examples:
        >>> from mindformers import AutoTokenizer
        >>> tokenize = AutoTokenizer.from_pretrained('glm2_6b')
        >>> tokenize("你好")
        {'input_ids': [64790, 64792, 36474, 54591], 'attention_mask': [1, 1, 1, 1]}
        >>> from mindformers import ChatGLM3Tokenizer
        >>> tokenizer = ChatGLM3Tokenizer('tokenizer.model')
        >>> prompts = ["晚上睡不着应该怎么办"]
        >>> token_id = tokenizer(prompts)
        >>> input_ids = token_id['input_ids']
        >>> print(input_ids)
        [[64790, 64792, 30910, 32820, 54266, 31876, 35153]]
        >>> response = tokenizer.decode(input_ids)
        >>> print(response)
        ['晚上睡不着应该怎么办']
    """
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    _support_list = MindFormerBook.get_tokenizer_support_list()['glm3']

    def __init__(self,
                 vocab_file,
                 bos_token='<sop>',
                 eos_token='<eop>',
                 end_token='</s>',
                 mask_token='[MASK]',
                 gmask_token='[gMASK]',
                 pad_token='<pad>',
                 unk_token='<unk>',
                 **kwargs):

        self.name = "ChatGLM3Tokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }

        self._bos_token = bos_token
        self._eos_token = eos_token
        self._end_token = end_token
        self._mask_token = mask_token
        self._gmask_token = gmask_token

        super().__init__(bos_token=bos_token,
                         eos_token=eos_token,
                         end_token=end_token,
                         mask_token=mask_token,
                         gmask_token=gmask_token,
                         pad_token=pad_token,
                         unk_token=unk_token,
                         **kwargs)

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token not in self.tokenizer.special_tokens:
            raise ValueError(f"{token} is not a special token for {self.name}.")
        return self.tokenizer.special_tokens[token]

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def build_single_message(self, role, metadata, message):
        if role not in ["system", "user", "assistant", "observation"]:
            raise ValueError(f"role should be one of 'system', 'user', 'assistant'"
                             f", 'observation', but got {role}.")
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens

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
        input_ids.extend([self.get_command("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors=return_tensors, is_split_into_words=True)

    def build_batch_input(self, queries, histories=None, roles="user", padding=True, return_tensors="np"):
        """build batch input with role."""
        if isinstance(queries, str):
            queries = [queries]
        batch_size = len(queries)
        if isinstance(roles, str):
            roles = [roles] * batch_size
        if isinstance(histories, list) and len(histories) != batch_size:
            histories = [histories]
        if histories is None:
            histories = [[] for _ in range(batch_size)]

        if batch_size != len(histories):
            raise ValueError(f"len(queries) should equal to len(histories), "
                             f"but got {len(queries) } and {len(histories)}.")
        if batch_size != len(roles):
            raise ValueError(f"len(queries) should equal to len(roles), "
                             f"but got {len(queries) } and {len(roles)}.")
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
            input_ids.extend([self.get_command("<|assistant|>")])
            batch_inputs.append(input_ids)

        return self.batch_encode_plus(batch_inputs, return_tensors=return_tensors,
                                      is_split_into_words=True, padding=padding)


    def tokenize(self, text, pair=None, add_special_tokens=True, **kwargs):
        """ Returns a tokenized string. """
        return self._tokenize(text)

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokenizer.convert_token_to_id(token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> list:
        """ Converts tokens to ids using the vocab. """
        ids = []
        for token in tokens:
            ids.append(self.tokenizer.convert_token_to_id(token))

        return ids

    def _decode(self,
                token_ids: Union[int, List[int]],
                skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = None,
                **kwargs) -> str:
        _, _ = skip_special_tokens, clean_up_tokenization_spaces
        tokens = []
        for token_id in token_ids:
            tokens.append(self._convert_id_to_token(token_id))
        return self.convert_tokens_to_string(tokens)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index >= self.vocab_size:
            raise IndexError(
                f"The token id {index} is out of the size of vocabulary, please check your tokenizer "
                f"and corresponding vocabulary files.")
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

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
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names.get("vocab_file")
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(vocab_file, flags_, 0o750), 'wb') as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

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
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

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

    # pylint: disable=W0221
    def apply_chat_template(self, conversation, return_tensors=None, **tokenizer_kwargs):
        if not conversation:
            return []
        if not isinstance(conversation, list) or len(conversation) != 1 or not isinstance(conversation[0], Dict):
            raise ValueError(f"conversation:{conversation} is invalid.")
        return self.build_chat_input(query=conversation[0].get("content"), role=conversation[0].get("role"),
                                     return_tensors=return_tensors)["input_ids"][0]
