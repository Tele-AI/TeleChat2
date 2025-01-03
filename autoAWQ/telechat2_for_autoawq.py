import tqdm
from typing import List, Tuple
from awq.models.base import BaseAWQForCausalLM,TRANSFORMERS_AUTO_MAPPING_DICT
from awq.quantize.quantizer import AwqQuantizer
import os
import gc
import json
import warnings
import logging
import torch
import transformers
import torch.nn as nn
from copy import deepcopy

from tqdm import tqdm
from typing import List, Union, Dict
from safetensors.torch import save_file
from typing_extensions import Doc, Annotated

from collections import defaultdict
import functools
import inspect

from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoProcessor,
    CLIPImageProcessor,
    PreTrainedTokenizer,
)

from awq.models._config import AwqConfig

from awq.quantize.quantizer import AwqQuantizer

TRANSFORMERS_AUTO_MAPPING_DICT['telechat'] = 'AutoModelForCausalLM'

class TeleChatV2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "TeleChatV2DecoderLayer"

    @staticmethod
    def get_model_layers(model):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)

    @staticmethod
    def get_layers_for_scaling(
            module, input_feat, module_kwargs
    ):
        layers = []
        module_kwargs_attenion = deepcopy(module_kwargs)
        module_kwargs_attenion["residual"] = input_feat["self_attention_residual"]
        layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                             module.self_attention.query,
                             module.self_attention.key_value
                            ],
                    inp=input_feat["self_attention"],
                    module2inspect=module.self_attention,
                    kwargs=module_kwargs_attenion,
                )
            )
        del module_kwargs_attenion
        clear_memory()
        ##GQA跳过缩放Output。
        # TODO:后面探索Output也做缩放的情况.
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        module_kwargs_mlp = deepcopy(module_kwargs)
        module_kwargs_mlp["residual"] = input_feat["mlp_residual"]
        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                kwargs=module_kwargs_mlp,
            )
        )
        del module_kwargs_mlp
        clear_memory()
        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                kwargs=module_kwargs,
            )
        )

        return layers

    @torch.no_grad()
    def quantize(
            self,
            tokenizer = None,
            quant_config = {},
            calib_data = "pileval",
            split = "train",
            text_column = "text",
            duo_scaling = True,
            export_compatible = False,
            apply_clip = True,
            n_parallel_calib_samples = None,
            max_calib_samples = 128,
            max_calib_seq_len = 512,
            max_chunk_memory = 1024 * 1024 * 1024,
    ):
        self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)

        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        self.quantizer = TeleChatAwqQuantizer(
            self,
            self.model,
            tokenizer,
            self.quant_config.w_bit,
            self.quant_config.q_group_size,
            self.quant_config.zero_point,
            self.quant_config.version,
            calib_data,
            split,
            text_column,
            duo_scaling,
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            export_compatible=export_compatible,
            apply_clip=apply_clip,
            n_parallel_calib_samples=n_parallel_calib_samples,
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory,
        )
        self.quantizer.quantize()

        self.is_quantized = True


class TeleChatAwqQuantizer(AwqQuantizer):
    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            if name in ['self_attention', 'mlp']:
                residual = x[1]
                if f'{name}_residual' not in feat_dict:
                    feat_dict[f'{name}_residual'] = []
                    feat_dict[f'{name}_residual'].append(residual)
                else:
                    feat_dict[f'{name}_residual'].append(residual)
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }
        if self.awq_model.model_type == "telechat":
            named_linears = {
                **named_linears,
                "self_attention": layer.self_attention,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat