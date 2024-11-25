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
"""Harness Eval"""
import copy
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Iterator

from tqdm import tqdm
import mindspore
from mindspore import Model, Tensor
from mindspore.common import initializer
import setproctitle
from lm_eval import utils
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator
)

from mindformers import (
    MindFormerConfig,
    build_context,
    build_parallel_config,
    AutoModel,
    MindFormerRegister
)
from mindformers.trainer.utils import transform_and_load_checkpoint

eval_logger = utils.eval_logger


@register_model("mf-auto", "mf", "mindformers")
class MFLM(TemplateLM):
    """
    An abstracted mindformers model class.

    Supports data-parallel multi-NPU.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    # pylint: disable=W0613
    def __init__(
            self,
            pretrained: str,
            use_past: Optional[bool] = False,
            device_id: Optional[int] = None,
            batch_size: Optional[int] = 1,
            max_length: Optional[int] = None,
            truncation: Optional[bool] = False,
            add_bos_token: Optional[bool] = False,
            prefix_token_id: Optional[int] = None,
            max_batch_size: Optional[int] = 64,
            use_parallel: Optional[bool] = False,
            dp: Optional[int] = 1,
            tp: Optional[int] = 1,
            **kwargs
    ) -> None:
        super().__init__()

        self.batch_size = int(batch_size)
        self._device = device_id
        self.batch_sizes = {}
        self.batch_schedule = 1
        self._max_length = max_length
        self.truncation = truncation
        self.add_bos_token = add_bos_token
        self.max_batch_size = max_batch_size

        model_config = self._get_config(
            pretrained=pretrained,
            batch_size=self.batch_size,
            use_parallel=use_parallel,
            use_past=use_past,
            dp=dp,
            tp=tp
        )

        self._create_model(model_config)
        self._create_tokenizer(pretrained=pretrained)

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return 1

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_config(
            self,
            pretrained: str,
            batch_size: int,
            use_parallel: bool = False,
            use_past: bool = False,
            tp: int = 1,
            dp: int = 1
    ) -> MindFormerConfig:
        """parse yaml configuration file"""
        # search yaml config file
        config_path = [str(file.resolve()) for file in Path(pretrained).glob('*.yaml')]
        if len(config_path) != 1:
            raise Exception("There is no or more than one config file in the model directory.")
        self._config = MindFormerConfig(config_path[0])

        self._config.pretrained = pretrained
        self._config.parallel_config.model_parallel = tp
        self._config.use_parallel = use_parallel
        self._config.parallel_config.data_parallel = dp
        self._config.auto_trans_ckpt = use_parallel

        if self._device:
            self._config.context.device_id = self._device
        else:
            self._device = self._config.context.device_id

        if self._max_length:
            self._config.processor.tokenizer.model_max_length = self._max_length
        else:
            self._max_length = self._config.processor.tokenizer.model_max_length

        if self._max_length:
            self._config.model.model_config.seq_length = self._max_length

        self._config.model.model_config.parallel_config = self._config.parallel_config
        self._config.model.model_config.batch_size = batch_size
        self._config.model.model_config.use_past = use_past

        build_context(self._config)
        build_parallel_config(self._config)

        return self._config

    def _create_model(self, config) -> None:
        """Initialize Model"""
        self._model = AutoModel.from_config(config)

        if not config.load_checkpoint:
            raise ValueError("There is no model ckpt in the model directory.")
        eval_logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        # set auto transform ckpt
        if config.use_parallel:
            config.auto_trans_ckpt = True
            eval_logger.info("----------------auto trans ckpt----------------")
        else:
            config.auto_trans_ckpt = False
        input_ids = Tensor(shape=(self.batch_size, seq_length), dtype=mindspore.int32, init=initializer.One())
        infer_data = self._model.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, Model(self._model), self._model, infer_data, do_predict=True)

    def _create_tokenizer(self, pretrained: str) -> None:
        """Initialize Tokenizer"""
        tokenizer_kwargs = dict(self.config.processor.tokenizer)
        tokenizer_type = tokenizer_kwargs.pop('type')
        try:
            tokenizer_class = MindFormerRegister.get_cls(module_type='tokenizer', class_name=tokenizer_type)
        except ValueError as e:
            tokenizer_py_path = [str(file.resolve()) for file in Path(pretrained).glob('*tokenizer*.py')]
            if len(tokenizer_py_path) != 1:
                raise Exception("There is no or more than one tokenizer python script in the model directory.") from e
            tokenizer_class = load_class_from_file(tokenizer_py_path[0], tokenizer_type)
        except Exception as e:
            raise e

        self.tokenizer = tokenizer_class(
            **tokenizer_kwargs
        )

    def tok_encode(
            self, string: str, left_truncate_len: Optional[int] = None, add_special_tokens=None
    ) -> List[int]:
        """encode tokens"""
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value

        # by default for CausalLM - false or self.add_bos_token is set

        if add_special_tokens is None:
            special_tokens_kwargs = {
                "add_special_tokens": False or self.add_bos_token
            }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            # pylint: disable=E1130
            encoding = encoding[-left_truncate_len:]

        return encoding

    # pylint: disable=E1130
    def tok_batch_encode(
            self,
            strings: List[str],
            padding_side: str = "left",
            left_truncate_len: Optional[int] = None,
            truncation: bool = False,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """encode tokens in batches"""
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="ms",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps):
        logits = self.model(inps)[0]
        return logits

    def _model_generate(self, context, max_length, **generation_kwargs):
        """model generate"""
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        generation_kwargs.pop('attention_mask')

        return self.model.generate(
            input_ids=context.tolist(),
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def _select_cont_toks(
            self, logits: mindspore.Tensor, contlen: int = None, inplen: int = None
    ) -> mindspore.Tensor:
        """select continuation tokens"""
        if not (contlen and inplen):
            raise ValueError("Must pass input len and cont. len to select scored logits for causal LM")
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen: inplen]

        return logits

    def loglikelihood_rolling(
            self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        """run task with loglikelihood_rolling"""
        loglikelihoods = []

        for (string,) in tqdm(
                [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _encode_pair(self, context, continuation):
        """encode contest and continuation"""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(
            self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """handle loglikelihood request type"""
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
            self,
            requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
            disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """handle loglikelihood_tokens request type"""
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            return req[-2] + req[-1][:-1]

        re_ord = Collator(requests, sort_fn=_collate, group_by="contexts", group_fn=_lookup_one_token_cont)

        chunks = re_ord.get_batched(self.batch_size)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                if not context_enc:
                    raise ValueError("context_enc must not be None")
                if not continuation_enc:
                    raise ValueError("continuation_enc must not be None")
                if len(continuation_enc) > self.max_length:
                    raise ValueError("The length of continuation_enc must be less than \
                        or equal to max_length, but got {}".format(len(continuation_enc)))

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = mindspore.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                    dtype=mindspore.int64
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")

            multi_logits = mindspore.ops.log_softmax(
                self._model_call(batched_inps), axis=-1
            )  # [batch, padding_length (inp or cont), vocab]
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                    chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (inplen + (logits.shape[0] - padding_len_inp))
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)

                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(axis=-1)
                eval_logger.info(f"answer:{self.tokenizer.decode(greedy_tokens)}")
                # eval_logger.info(f"{self.tokenizer.decode(res)}")

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str_, cont_toks_, logits_ in get_cache(
                        self=re_ord,
                        req_str=request_str,
                        cxt_toks=ctx_tokens,
                        cont_toks=cont_toks,
                        logits=logits,
                ):
                    cont_toks_ = mindspore.tensor(
                        cont_toks_, dtype=mindspore.int64
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks_).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits_ = mindspore.mint.gather(logits_, 2, cont_toks_.unsqueeze(-1)).squeeze(-1)

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits_.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str_, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
            self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """handle generate_until request type"""
        res = []

        def _collate(req: Tuple[str, dict]):
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        chunks = re_ords.get_batched(self.batch_size)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode([self.eot_token_id], skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            max_ctx_len = self.max_length - max_gen_toks

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                **kwargs,
            )

            for cont_toks, context in zip(cont, contexts):
                cont_toks = cont_toks[context_enc.shape[1]:]

                s = self.tok_decode(cont_toks)

                for term in until:
                    s = s.split(term)[0]
                eval_logger.info(f"\n\n<answer>\n{s}\n")
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    def get_model_info(self) -> dict:
        """get model info"""

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
        }

        return model_info


def load_class_from_file(module_path, class_name):
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, class_name)


def pad_and_concat(
        max_length: int,
        tensors: List[mindspore.Tensor],
        padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'")

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = mindspore.ops.cat(
                    [
                        tensor,
                        mindspore.ops.zeros(
                            max_length - tensor_len,
                            dtype=mindspore.int64,
                        ),
                    ],
                    axis=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = mindspore.ops.cat(
                    [
                        mindspore.ops.zeros(
                            max_length - tensor_len,
                            dtype=mindspore.int64,
                        ),
                        tensor,
                    ],
                    axis=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return mindspore.ops.cat(tensors, axis=0)


# pylint: disable=W0212
def get_cache(
        self,
        req_str: Tuple[str, str] = None,
        cxt_toks: List[int] = None,
        cont_toks: List[int] = None,
        logits: mindspore.Tensor = None,
) -> Iterator[Tuple[Tuple[str, str], List[int], mindspore.Tensor]]:
    """get requests cache"""
    if self._group_by == "contexts":
        cache_hit: List[
            Tuple[int, Tuple[Tuple[str, str], List[int], List[int]]]
        ] = self._arr_with_indices.pop(tuple(cxt_toks + cont_toks[:-1]))
        cache_size = len(cache_hit)
        if cache_size == 1:
            self._reorder_indices.extend(x[0] for x in cache_hit)
            yield req_str, cont_toks, logits
        else:
            # If we have matching requests then expand the batch dimension (no-op) and
            # yield each along with its corresponding args.
            multilogits = logits.broadcast_to((cache_size, -1, -1)).chunk(cache_size)
            indices, req_str, cont_toks = zip(
                *[(x[0], x[1][0], x[-1][-1]) for x in cache_hit]
            )
            self._reorder_indices.extend(indices)
            for c_key, cont_tok, logit in zip(req_str, cont_toks, multilogits):
                yield c_key, cont_tok, logit
    else:
        yield req_str, cont_toks, logits


if __name__ == '__main__':
    setproctitle.setproctitle("ms_main_thread")
    cli_evaluate()
