import torch
from torch import nn
from typing import List, Dict
from awq.models.base import BaseAWQForCausalLM, TRANSFORMERS_AUTO_MAPPING_DICT
from awq.quantize.quantizer import AwqQuantizer

from awq.models._config import AwqConfig

TRANSFORMERS_AUTO_MAPPING_DICT["telechat"] = "AutoModelForCausalLM"


class OldTelechat2Block:
    """
    Just as a type description, consistent with the modeling_telechat.py
    """

    ...


class OldTelechatForCausalLM:
    """
    Just as a type description, consistent with the modeling_telechat.py
    """

    ...


class Telechat2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "TelechatBlock"
    max_seq_len_key = "training_seqlen"
    modules_to_not_convert = ["query", "key_value", "dense"]

    @staticmethod
    def fuse_layers(model: OldTelechatForCausalLM):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldTelechatForCausalLM):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module: OldTelechat2Block):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldTelechatForCausalLM, device: str):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldTelechat2Block, input_feat, module_kwargs):
        """
        Due to the unique organization of key value weights in telechat2,
        the qkv and dense layers cannot be quantified
        """
        layers = []

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                kwargs={"residual": input_feat["post_attention_layernorm"]},
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


class Telechat2AwqQuantizer(AwqQuantizer):
    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        # Inorder to get device
        assert len(layers) > 0
        device = next(layers[0].parameters()).device
        # Put other tensor on the right device, e.g post_attention_layernorm
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device)
        res = super()._search_best_scale(
            module=module,
            prev_op=prev_op,
            layers=layers,
            inp=inp,
            module2inspect=module2inspect,
            kwargs=kwargs,
        )
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key].cpu()
        return res

    def _get_input_feat(
        self, layer: OldTelechat2Block, named_linears: Dict[str, nn.Linear]
    ):
        if self.awq_model.model_type == "telechat":
            named_linears = {
                **named_linears,
                "post_attention_layernorm": layer.post_attention_layernorm,
            }
        return super()._get_input_feat(layer=layer, named_linears=named_linears)


__all__ = ["Telechat2AWQForCausalLM", "Telechat2AwqQuantizer"]
