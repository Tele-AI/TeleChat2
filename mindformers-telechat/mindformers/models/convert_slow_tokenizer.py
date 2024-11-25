# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings
from typing import Dict, List, Tuple
import importlib.util
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def import_protobuf(error_message=""):
    """import_protobuf"""
    if is_protobuf_available():
        import google.protobuf

        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            from mindformers.models import sentencepiece_model_pb2
        else:
            from mindformers.models import sentencepiece_model_pb2_new as sentencepiece_model_pb2
        return sentencepiece_model_pb2
    raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    """BertConverter"""
    def converted(self) -> Tokenizer:
        """BertConverter's converted"""
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class GPT2Converter(Converter):
    """GPT2Converter"""
    def converted(self) -> Tokenizer:
        """GPT2Converter's converted"""
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if self.original_tokenizer.add_bos_token:
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class SpmConverter(Converter):
    """SpmConverter"""
    def __init__(self, *args):
        super().__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback "
                    "option which is not implemented in the fast tokenizers. In practice this means that the fast "
                    "version of the tokenizer can produce unknown tokens whereas the sentencepiece version would have "
                    "converted these unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        """SpmConverter's tokenizer"""
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if not precompiled_charsmap:
            return normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " ")])
        return normalizers.Sequence(
            [normalizers.Precompiled(precompiled_charsmap), normalizers.Replace(Regex(" {2,}"), " ")]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    # # pylint: disable=E1128
    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


class T5Converter(SpmConverter):
    """T5Converter"""

    # pylint: disable=W0212
    def vocab(self, proto):
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class CLIPConverter(Converter):
    """CLIPConverter"""
    def converted(self) -> Tokenizer:
        """CLIPConverter's converted"""
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        # Hack to have a ByteLevel and TemplaceProcessor
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )
        return tokenizer


class LlamaConverter(SpmConverter):
    """LlamaConverter"""
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            import tokenizers

            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def post_processor(self):
        # the processor is defined in the LlamaTokenizerFast class.
        return None



SLOW_TO_FAST_CONVERTERS = {
    "BertTokenizer": BertConverter,
    "CLIPTokenizer": CLIPConverter,
    "GPT2Tokenizer": GPT2Converter,
    "T5Tokenizer": T5Converter,
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast converters:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()
