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
"""Create tokenizer."""
import os
import shutil
import json
import gzip
import tokenizers
from tokenizers import (
    AddedToken,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import sentencepiece as spm


string = """
        华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。
        An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.
        """


def get_sp_vocab_model(model_type, model_path):
    """Get sp vocab model."""
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            corpus_path = os.path.join(model_path, "corpus.txt")
            res_path = os.path.join(model_path, f"{model_type}_tokenizer")
            with open(corpus_path, "w", encoding="utf-8") as w:
                w.write(string)

            spm.SentencePieceTrainer.Train(
                input=corpus_path,
                model_prefix=res_path,
                vocab_size=200,
                character_coverage=1, model_type='bpe', num_threads=8
            )
            retry = False
            success_sig = True
        # pylint: disable=W0703
        except Exception as e:
            corpus_path = os.path.join(model_path, "corpus.txt")
            res_path = os.path.join(model_path, f"{model_type}_tokenizer")
            if os.path.exists(corpus_path):
                shutil.rmtree(corpus_path)
            if os.path.exists(f"{res_path}.model"):
                shutil.rmtree(f"{res_path}.model")
            if os.path.exists(f"{res_path}.vocab"):
                shutil.rmtree(f"{res_path}.vocab")
            print(f"{model_type} tokenizer model initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"{model_type} tokenizer initialize failed for {count} times.")


def get_gpt_vocab_model(model_path):
    """Get gpt vocab model."""
    tokens = ['Ġthree', 'çļĦ', 'åį', 'å®', 'åħ', '.', 'Ġtwo', 'åľ', 'Ġsequence', 'æ', 'Ġincreasing', '¸', ',', 'ä¸Ń',
              'å¤', '§', 'ĸ', 'ĳ', 'Ġone', 'An', '¬', ':', 'åĽ', '½', '±', 'į', 'ĥ', 'åĮ', 'ä¸Ģ', 'ç', 'éĥ', 'İ', '³',
              '¶æ', 'æĺ¯', '¨', 'åı', 'º', 'ļ', '»', 'Ĭ', 'Ģ', 'ãĢĤ', 'äº', '·', 'ä¸', 'ä½']
    merges = ['. .', '. "', ', "', ': //', '. ,', '. )', ". '", '. [', ': :', ", '", 'Ġincreasing ly', 'An other',
              'An y', ': \\', '. -', '. ]', '. --', ', ,', ': "', '. :', ', -', '. ),', '. (', '. âĢĶ', '. _', ': /',
              '. ;', 'ç ¥ŀ', '. ).', '. *', 'æ ľ', ', [', 'ç Ķ', 'An n', 'ç ļ', '. /', 'An onymous', 'ļ é', 'äº º',
              ': -', 'æ Ī', 'İ ĭ', 'æ ĺ', 'æ ĸ', 'å¤ ©', 'ç İĭ', '. </', ': {', 'æ Ń', '. <', 'æ Ŀ', 'An th', 'An na',
              'ĸ ļ', 'ä¸ Ģ', 'ç ī', 'å¤ §', ": '", '. #', 'An alysis', '. >>', ': [', 'ç Ľ', 'æ ī', 'An swer', 'æ Ĺ',
              ', âĢĶ', 'An im', 'æ Ħ', ', )', 'æ °', '» Ĵ', 'æ ³', ': (', '. âĢĵ', 'An aly', 'æ µ', ', .', 'ä¸ į',
              'ĳ å£«', 'An imation', 'An imal', 'ç ľ', '. ï¿½', '. ?', 'ä¸ Ń', 'ä¸ Ĭ', ', ...', 'An ne', 'ä½ ľ',
              'æ ł', '. ãĢį', 'æ ©', ': #', 'ç «', '. ,"', '. }', 'An cient', 'æ Ģ', ': ,', 'ç ĭ', '¬ ¼', 'ä½ ¿',
              'ç ·', 'ç ķ', 'åħ ī', 'æ ĥ', 'æ ķ', 'Ĭ ±', 'ç Ħ', ': ]', '. $', 'ä¸ ī', 'äº Ķ', 'ç Ĳ']

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            vocab_path = os.path.join(model_path, "gpt_vocab.json")
            with open(vocab_path, "w", encoding="utf-8") as w:
                cnt = 0
                vocab_json = {}
                for token in tokens:
                    vocab_json[token] = cnt
                    cnt += 1
                w.write(json.dumps(vocab_json))
            merges_path = os.path.join(model_path, "gpt_merges.txt")
            with open(merges_path, "w", encoding="utf-8") as w:
                for merge in merges:
                    w.write(merge + "\n")
            retry = False
            success_sig = True
        # pylint: disable=W0703
        except Exception as e:
            if os.path.exists(vocab_path):
                shutil.rmtree(vocab_path)
            if os.path.exists(merges_path):
                shutil.rmtree(merges_path)
            print(f"gpt tokenizer model initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

        if not success_sig:
            raise RuntimeError(f"gpt tokenizer initialize failed for {count} times.")


def get_bbpe_vocab_model(model_type, model_path):
    """Get bpe vocab model."""
    if model_type.startswith("bert"):
        tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    else:
        tokens = ["<unk>", "<s>", "</s>", "<pad>"]

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            special_tokens = []
            for token in tokens:
                special_tokens.append(AddedToken(content=token, special=True))

            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Split(
                    pattern=tokenizers.Regex(" ?[^(\\s|[.,!?…。，、।۔،])]+"), behavior="isolated", invert=False),
                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
            )
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

            trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=special_tokens)
            tokenizer.train_from_iterator([string], trainer=trainer)
            tokenizer.decoder = decoders.ByteLevel()

            vocab_tmp_path = os.path.join(model_path, f"{model_type}_tmp_tokenizer.json")
            vocab_path = os.path.join(model_path, f"{model_type}_tokenizer.json")
            tokenizer.save(vocab_tmp_path)
            with open(vocab_tmp_path, "r") as r, open(vocab_path, "w") as w:
                all_json = json.loads(r.read())
                all_json["model"]["merges"] = [" ".join(item) if isinstance(item, list) else item \
                                               for item in all_json["model"]["merges"]]
                w.write(json.dumps(all_json))
            if model_type == "gpt":
                vocab_json_path = os.path.join(model_path, f"{model_type}_vocab.json")
                merges_path = os.path.join(model_path, f"{model_type}_merges.txt")
                with open(vocab_json_path, "w") as w_json, \
                        open(merges_path, "w") as w_merges:
                    w_json.write(json.dumps(all_json["model"]["vocab"]))
                    w_merges.write("\n".join(all_json["model"]["merges"]))
            if model_type == "clip":
                vocabtxt_path = os.path.join(model_path, f"bpe_simple_vocab_16e6.txt")
                with open(vocabtxt_path, "w") as w_merges:
                    w_merges.write("\n".join(all_json["model"]["merges"]))
                gzip_path = os.path.join(model_path, f"bpe_simple_vocab_16e6.txt.gz")
                # 使用gzip打开目标文件，并使用shutil.copyfileobj复制内容
                with gzip.open(gzip_path, 'wb') as gz_file:
                    with open(vocabtxt_path, 'rb') as f:
                        shutil.copyfileobj(f, gz_file)
            retry = False
            success_sig = True
        # pylint: disable=W0703
        except Exception as e:
            vocab_path = os.path.join(model_path, f"{model_type}_tokenizer.json")
            if os.path.exists(vocab_path):
                shutil.rmtree(vocab_path)
            if model_type == "gpt":
                vocab_path = os.path.join(model_path, f"{model_type}_vocab.json")
                merges_path = os.path.join(model_path, f"{model_type}_merges.txt")
                if os.path.exists(vocab_path):
                    shutil.rmtree(vocab_path)
                if os.path.exists(merges_path):
                    shutil.rmtree(merges_path)
            if model_type == "clip":
                vocabtxt_path = os.path.join(model_path, f"bpe_simple_vocab_16e6.txt")
                gzip_path = os.path.join(model_path, f"bpe_simple_vocab_16e6.txt.gz")
                if os.path.exists(vocabtxt_path):
                    shutil.rmtree(vocabtxt_path)
                if os.path.exists(gzip_path):
                    shutil.rmtree(gzip_path)
            print(f"{model_type} tokenizer model initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False
    if not success_sig:
        raise RuntimeError(f"{model_type} tokenizer initialize failed for {count} times.")


def get_wordpiece_vocab_model(model_path):
    """Get wordpiece vocab model."""
    retry = True
    count = 0
    success_sig = False
    corpus_path = os.path.join(model_path, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as w:
        w.write(string)
    while retry:
        try:
            count += 1
            tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(),
                 normalizers.Lowercase(),
                 normalizers.StripAccents()]
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.WhitespaceSplit(),
                 pre_tokenizers.Punctuation()]
            )
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            trainer = trainers.WordPieceTrainer(vocab_size=200, special_tokens=special_tokens)
            tokenizer.train_from_iterator([string], trainer=trainer)
            cls_token_id = tokenizer.token_to_id("[CLS]")
            sep_token_id = tokenizer.token_to_id("[SEP]")
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"[CLS]:0 $A:0 [SEP]:0",
                pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
            )
            tokenizer.decoder = decoders.WordPiece(prefix="##")
            vocab_json_path = os.path.join(model_path, "bert_tokenizer.json")
            vocab_txt_path = os.path.join(model_path, "bert_vocab.txt")
            tokenizer.save(vocab_json_path)
            with open(vocab_json_path, "r") as r, open(vocab_txt_path, "w") as w:
                all_json = json.loads(r.read())
                print("========", all_json["model"]["vocab"])
                all_json["model"]["vocab"] = sorted(list(all_json["model"]["vocab"].keys()))

                w.write("\n".join(all_json["model"]["vocab"]))
            retry = False
            success_sig = True
        # pylint: disable=W0703
        except Exception as e:
            vocab_path = os.path.join(model_path, "bert_tokenizer.json")
            vocab_txt_path = os.path.join(model_path, "bert_vocab.txt")
            if os.path.exists(vocab_path):
                shutil.rmtree(vocab_path)
            if os.path.exists(vocab_txt_path):
                shutil.rmtree(vocab_txt_path)
            print(f"bert tokenizer model initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False
    if not success_sig:
        raise RuntimeError("bert tokenizer initialize failed for {count} times.")
