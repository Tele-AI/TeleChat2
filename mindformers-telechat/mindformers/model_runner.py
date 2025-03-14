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

"""
For text generation
"""
import os
import json
from typing import Optional, List, Union, Dict
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.common.initializer import Zero
from mindspore._c_expression import swap_cache
from mindspore.communication.comm_func import barrier

from mindformers import models, MindFormerRegister, MindFormerModuleType
from mindformers import build_context, build_parallel_config, GenerationConfig
from mindformers import AutoModel, AutoConfig, AutoTokenizer
from mindformers.models.utils import convert_mstype, str_to_ms_type
from mindformers.utils import contains_safetensors_files, is_hf_safetensors_dir, process_hf_checkpoint

from mindformers.tools.logger import logger
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.tools.hub.dynamic_module_utils import get_class_from_dynamic_module
from mindformers.generation.parallel_decoding import parallel_decoding_control
from mindformers.version_control import check_delay_init_valid, need_nz
from mindformers.models import build_processor, PretrainedConfig

__all__ = ["ModelRunner"]


def register_auto_class(config, pretrained_model_name_or_path, class_type, use_fast=True):
    """convert to auto class"""
    if config.model.model_config.auto_map:
        class_auto = config["model"]["model_config"]["auto_map"]
        if class_type == "AutoConfig" and \
                config.model.model_config.type not in MindFormerRegister.registry[MindFormerModuleType.CONFIG]:
            class_ref = class_auto[class_type]
            config_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(config_class, module_type=MindFormerModuleType.CONFIG)

        if class_type == "AutoTokenizer" and \
                config.processor.tokenizer.type not in MindFormerRegister.registry[MindFormerModuleType.TOKENIZER]:
            if use_fast and class_auto[class_type][1] is not None:
                class_ref = class_auto[class_type][1]
            else:
                class_ref = class_auto[class_type][0]
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(tokenizer_class, module_type=MindFormerModuleType.TOKENIZER)

        if class_type == "AutoModel" and \
                config.model.arch.type not in MindFormerRegister.registry[MindFormerModuleType.MODELS]:
            class_ref = class_auto[class_type]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(model_class, module_type=MindFormerModuleType.MODELS)

        if class_type == "AutoProcessor" and \
                config.model.arch.type not in MindFormerRegister.registry[MindFormerModuleType.PROCESSOR]:
            class_ref = class_auto[class_type]
            processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(processor_class, module_type=MindFormerModuleType.PROCESSOR)


def is_multi_modal_model(config):
    def count_type_num(model_config):
        num = 0
        for k, v in model_config.items():
            if k == "type":
                num += 1
            if isinstance(v, dict):
                num += count_type_num(v)
        return num
    return count_type_num(config.model.model_config) > 1


def get_model(model_name_or_path: str,
              revision: Optional[str] = None,
              trust_remote_code: Optional[bool] = False,
              **kwargs):
    """
    get_model API, supports MF to be a backend of MindIEServer.

    Args:
        model_name_or_path (str):
            A path to a *directory* containing vocabulary files() required by the tokenizer.
        revision (`str`, *optional*, defaults to `"None"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id.
        trust_remote_code (`bool`, *optional*, defaults to `True`):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional key word arguments for AutoTokenizer.from_pretrained.

    Returns:
        A Tokenizer object and others.
    """
    if not os.path.exists(model_name_or_path) or not os.path.isdir(model_name_or_path):
        raise ValueError(f"{model_name_or_path} does not exist or is not a directory.")

    logger.debug(f"model_name_or_path is {model_name_or_path}")
    config_path = _get_model_config(model_name_or_path)
    config = MindFormerConfig(config_path)
    model_type = config.model.arch.type
    logger.info(f"The model type is: {model_type}")
    register_auto_class(config, model_name_or_path, class_type="AutoTokenizer")

    if is_multi_modal_model(config):
        processor = build_processor(config.processor)
        return processor, processor

    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision,
                                              trust_remote_code=trust_remote_code,
                                              use_fast=use_fast)

    input_builder = InputBuilder(tokenizer)
    return tokenizer, input_builder


class ModelRunner:
    """
    ModelRunner API, supports MindFormers to be a backend of MindIEServer.

    Args:
        model_path (str):
            The model config path contains model config file and tokenizer file.
        npu_mem_size (int):
            Npu memory size used for kv-cache.
        cpu_mem_size (int):
            Cpu memory size used for kv-cache.
        block_size (int):
            Block size used for kv-cache.
        rank_id (int, optional):
            Rank id used for infer. Default: ``0``.
        world_size (int, optional):
            Rank size used for infer. Default: ``1``.
        npu_device_ids (list[int], optional):
            Get npu_device_ids from MindIE config. Default: ``None``.
        plugin_params (str, optional):
            A JSON string that contains additional plugin parameters. Default: ``None``.

    Returns:
        A MindIERunner object.

    Examples:
        >>> from mindformers import ModelRunner
        >>> model_path = /path/to/model/ # contains model config file and tokenizer file.
        >>> npu_mem_size = 3
        >>> cpu_mem_size = 1
        >>> block_size = 128
        >>> rank_id = 0
        >>> world_size = 1
        >>> npu_device_ids = [0]
        >>> model_runner = ModelRunner(model_path=model_path, npu_mem_size=npu_mem_size, cpu_mem_size=cpu_mem_size,
        >>>                            block_size=block_size, rank_id=rank_id, world_size=world_size,
        >>>                            npu_device_ids=npu_device_ids)
        >>> type(model_runner)
        <class 'mindformers.model_runner.MindIEModelRunner'>
    """

    def __new__(cls, model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                npu_device_ids=None, plugin_params=None):
        config_path = _get_model_config(model_path)
        config = MindFormerConfig(config_path)
        model_type = config.model.arch.type
        logger.info(f"The model type is: {model_type}")
        model_runner_cls = MindIEModelRunner
        if model_type not in models.__all__:
            try:
                import importlib
                model_runner_cls = importlib.import_module(model_type, ["MindIEModelRunner"]).MindIEModelRunner
            except ImportError:
                logger.info(f"import MindIEModelRunner from module {model_type} failed, "
                            f"and will use the default one defined in mindformers.")

        model_runner = model_runner_cls(model_path, config_path, npu_mem_size, cpu_mem_size,
                                        block_size, rank_id, world_size, npu_device_ids, plugin_params)
        return model_runner


class MindIEModelRunner:
    """
    Implementation of ModelRunner.

    Args:
        model_path(str):
            The model config path contains model config file and tokenizer file.
        experiment_mode (bool):
            Is experiment model.
        model_config (PretrainedConfig):
            Model config.
        npu_mem_size (int):
            Npu memory size used for kv-cache.
        cpu_mem_size (int):
            Cpu memory size used for kv-cache.
        block_size (int):
            Block size used for kv-cache.
        rank_id (int):
            Rank id used for infer.
        world_size (int):
            Rank size used for infer.
        npu_device_ids (list[int]):
            Get npu_device_ids from MindIE config.
        plugin_params (str):
            A JSON string that contains additional plugin parameters.
    """

    def __init__(self, model_path, config_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0,
                 world_size=1, npu_device_ids=None, plugin_params=None):
        self.dynamic_kv_cache_whitelist = ["ParallelLlamaForCausalLM"]
        self.config = MindFormerConfig(config_path)
        self.warmup_step = 2
        self.is_multi_modal_model = is_multi_modal_model(self.config)
        # register to Auto Class
        register_auto_class(self.config, model_path, class_type="AutoConfig")
        register_auto_class(self.config, model_path, class_type="AutoTokenizer")
        register_auto_class(self.config, model_path, class_type="AutoModel")

        # parallel predict with dynamic cluster.
        if world_size > 1:
            self.config.use_parallel = True
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
            os.environ['MS_NODE_ID'] = str(rank_id)
            ms.set_device("Ascend", npu_device_ids[0])
            if rank_id == 0 and os.fork() == 0:
                os.environ['MS_ROLE'] = 'MS_SCHED'
                init()
        if self.config.use_parallel:
            build_parallel_config(self.config)
            self.model_config = AutoConfig.from_pretrained(config_path, parallel_config=self.config.parallel_config)
        else:
            self.model_config = AutoConfig.from_pretrained(config_path)
        setattr(self.model_config, 'npu_mem_size', npu_mem_size)
        self.use_legacy = getattr(self.model_config, "use_legacy", True)
        if self.config.moe_config:
            self.model_config.moe_config = self.config.moe_config

        self.update_model_config(plugin_params)

        if self.is_multi_modal_model:
            if isinstance(self.model_config.llm_model, PretrainedConfig):
                llm_config = self.model_config.llm_model
            else:
                llm_config = self.model_config.llm_model.model_config
            self.update_llm_config(llm_config, world_size, npu_mem_size, cpu_mem_size, block_size)
            self.processor = build_processor(self.config.processor)
            # adapt to mindie-llm
            self.model_config.max_position_embedding = llm_config.max_position_embedding
        else:
            self.update_llm_config(self.model_config, world_size, npu_mem_size, cpu_mem_size, block_size)

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        if not self.config.use_parallel and npu_device_ids:
            if len(npu_device_ids) != 1:
                raise ValueError("npu_device_ids should only contain one device_id")
            self.config.context.device_id = npu_device_ids[0]

        build_context(self.config)
        logger.info(f"Build context finished.")

        # build tokenizer
        if self.is_multi_modal_model:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
        logger.info(f"Build tokenizer finished.")

        # build model
        network_delay_inited = False
        if check_delay_init_valid():
            from mindspore.nn.utils import no_init_parameters
            with no_init_parameters():
                self.model = AutoModel.from_config(self.model_config)
            network_delay_inited = True
            logger.info("Parameters are not initialized during model initialization.")
        else:
            self.model = AutoModel.from_config(self.model_config)
        if npu_mem_size == -1 and str(type(self.model).__name__) not in self.dynamic_kv_cache_whitelist:
            raise ValueError("npu_mem_size=-1 only support in parallel mode")
        logger.info(f"Build model finished.")

        self.load_checkpoint(network_delay_inited)

        if self.model_config.is_dynamic:
            self.model.set_dynamic_inputs()

        cpu_kv_shape = (self.cpu_num_blocks, block_size, self.num_kv_heads, self.head_size)
        self.key_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=self.dtype, init=Zero()),
                                      name=f"key_host_{i}", requires_grad=False) for i in range(self.num_layers)]
        self.value_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=self.dtype, init=Zero()),
                                        name=f"value_host_{i}", requires_grad=False) for i in range(self.num_layers)]

    def load_checkpoint(self, network_delay_inited):
        """load checkpoint into model"""
        ms_model = ms.Model(self.model)
        batch_size = self.model_config.batch_size
        seq_length = self.model_config.seq_length
        input_ids = np.ones(shape=tuple([batch_size, seq_length]))
        if self.use_legacy:
            inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
        else:
            inputs = None
        load_checkpoint = self.config.load_checkpoint
        if (self.config.get('load_ckpt_format', 'ckpt') == 'safetensors'
                and is_hf_safetensors_dir(load_checkpoint, self.model)):
            self.config.load_checkpoint = process_hf_checkpoint(self.model, self.config.output_dir, load_checkpoint)
            if self.config.use_parallel:
                barrier()
        transform_and_load_checkpoint(self.config, ms_model, self.model, inputs, do_predict=True)
        if network_delay_inited:
            self.model.init_parameters_data()
        logger.info(f"Load checkpoints finished.")

    def update_model_config(self, plugin_params):
        """update model config"""
        self.model_config.parallel_decoding_params = None
        default_plugin_configs = {'plugin_type': None}
        if plugin_params == default_plugin_configs:
            plugin_params = None
        if plugin_params:
            if not isinstance(plugin_params, dict):
                plugin_params = json.loads(plugin_params)
            plugin_params['parallel_decoding'] = plugin_params['plugin_type']
            self.model_config.parallel_decoding_params = plugin_params
        self.model_config.checkpoint_name_or_path = None
        self.model_config.checkpoint_path = self.config.load_checkpoint

    def update_llm_config(self, config, world_size, npu_mem_size, cpu_mem_size, block_size):
        """update llm model config"""
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_heads if config.n_kv_heads is None \
            else config.n_kv_heads
        self.num_kv_heads = self.num_kv_heads // world_size  # check the divisibility in model initialization.
        self.head_size = config.hidden_size // config.num_heads

        kvcache_dtype = config.compute_dtype
        if hasattr(self.model_config, "quantization_config") and \
                self.model_config.quantization_config.kvcache_dtype in str_to_ms_type:
            kvcache_dtype = self.model_config.quantization_config.kvcache_dtype
        self.dtype = convert_mstype(kvcache_dtype)
        kvcache_bytes = ms.Tensor(0, dtype=self.dtype).itemsize

        total_head_size = self.num_kv_heads * self.head_size
        if need_nz():
            total_head_size = -(total_head_size // -16) * 16
        self.npu_num_blocks = (npu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * total_head_size * kvcache_bytes * 2 * self.num_layers)
        self.cpu_num_blocks = (cpu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * total_head_size * kvcache_bytes * 2 * self.num_layers)
        config.block_size = block_size
        config.num_blocks = self.npu_num_blocks

        if not hasattr(config, "max_position_embedding") or not config.max_position_embedding:
            config.max_position_embedding = config.seq_length

    def forward(self, input_ids: [Union[List[int], List[List[int]]]],
                valid_length_each_example: List[int],
                block_tables: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                prefill: bool = True,
                position_ids: Optional[Tensor] = None,
                spec_mask: Optional[Tensor] = None,
                q_seq_lens: Optional[Tensor] = None,
                adapter_ids: Optional[List[str]] = None,
                prefill_head_indices: Optional[Tensor] = None,
                key_cache: Optional[List[Tensor]] = None,
                value_cache: Optional[List[Tensor]] = None):
        """
        Call self.model.infer() or self.model.forward() to do infer and return logits on next position, \
        can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            block_tables (Tensor):
                Params for page attention
            slot_mapping (Tensor):
                Params for page attention
            prefill (bool):
                Whether to do prefill predict or decode predict
            position_ids (Tensor):
                Params for position encoding
            spec_mask (Tensor):
                Params for page attention
            q_seq_lens (Tensor):
                Params for page attention
            adapter_ids (List(str)):
                Params for SLora request
            prefill_head_indices (Tensor):
                Params for pre gather
            key_cache (List(Tensor), optional):
                Params for key_cache, a group of tensors used for kvcache. Default: None.
            value_cache (List(Tensor), optional):
                Params for value_cache, a group of tensors used for kvcache. Default: None.

        Returns:
            logits (Tensor)
        """
        is_warm_up = self.warmup_step > 0
        valid_length_each_example = np.array(valid_length_each_example)
        model_args = {"mindie_warm_up": is_warm_up}

        if self.is_multi_modal_model and not is_warm_up:
            if prefill:
                input_ids, decode_args = self.processor.decode_input_ids(input_ids, valid_length_each_example)
                decode_args.pop("position_ids", None)
                model_args.update(decode_args)

        if self.use_legacy:
            res, current_idx = self.model.forward(input_ids=input_ids,
                                                  valid_length_each_example=valid_length_each_example,
                                                  block_tables=block_tables,
                                                  slot_mapping=slot_mapping,
                                                  prefill=prefill,
                                                  use_past=True,
                                                  position_ids=position_ids,
                                                  spec_mask=spec_mask,
                                                  q_seq_lens=q_seq_lens,
                                                  adapter_ids=adapter_ids,
                                                  prefill_head_indices=prefill_head_indices,
                                                  key_cache=key_cache,
                                                  value_cache=value_cache,
                                                  **model_args)
        else:
            res, current_idx = self.model.forward_mcore(input_ids=input_ids,
                                                        valid_length_each_example=valid_length_each_example,
                                                        block_tables=block_tables,
                                                        slot_mapping=slot_mapping,
                                                        prefill=prefill,
                                                        position_ids=position_ids,
                                                        spec_mask=spec_mask,
                                                        q_seq_lens=q_seq_lens,
                                                        adapter_ids=adapter_ids,
                                                        prefill_head_indices=prefill_head_indices,
                                                        key_cache=key_cache,
                                                        value_cache=value_cache,
                                                        **model_args)
        logits = res[0] if isinstance(res, tuple) else res
        if hasattr(self, 'model_config') and parallel_decoding_control(self.model_config):
            return logits
        if self.use_legacy and prefill and logits.shape[0] > len(current_idx):
            logits = logits[Tensor(current_idx)]

        if self.warmup_step > 0:
            self.warmup_step -= 1
        return logits

    def swap(self, block_tables, swap_type):
        """
        Swap key/value cache between host and device, to support multi-batch and long-sequence inference.

        Args:
            block_tables:
                A 2-D array contains src and dst blocks to swap.
            swap_type:
                A bool value indicating the data direction: "True" for device-to-host, and "False" for host-to-device.
        """
        for i in range(self.num_layers):
            key_cache, value_cache = self.model.kvcache(i)
            swap_cache(self.key_host[i], key_cache, ms.Tensor(block_tables), swap_type)
            swap_cache(self.value_host[i], value_cache, ms.Tensor(block_tables), swap_type)

    def generate_position_ids(self, input_ids):
        if not self.is_multi_modal_model or self.warmup_step > 0:
            return range(len(input_ids))
        return self.processor.decode_position_ids_from_input_ids(input_ids)


def _get_model_config(model_path):
    """
    Get model config from the config file.

    Args:
        model_path: path of model config file.

    Returns:
        config_path.
    """

    if os.path.isdir(model_path):
        yaml_list = [file for file in os.listdir(model_path)
                     if file.endswith(".yaml")]
        if yaml_list:
            yaml_path = os.path.join(model_path, yaml_list[0])
        else:
            raise FileNotFoundError(f"There is no yaml file for model config in {model_path}.")
    else:
        raise ValueError(f"The path {model_path} is not exist.")

    return yaml_path


class InputBuilder:
    """
    Implementation of InputBuilder.

    Args:
        tokenizer (PreTrainedTokenizer):
            A tokenizer for text processing.
        chat_template (str):
            A Jinja template to use for this conversion.
        system_role_name (str):
            The name of system role.
        user_role_name (str):
            The name of user role.
        max_length (int):
            The max length of input tokens.
    """

    def __init__(self, tokenizer, chat_template="", system_role_name="system", user_role_name="user", max_length=2048):
        self.tokenizer = tokenizer
        self.system_role_name = system_role_name
        self.user_role_name = user_role_name
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.max_length = max_length
        self.rank = 0
        self.adapt_to_max_length = False

    def make_context(self, rank: int, conversation: List[Dict[str, str]], add_generation_prompt: bool = True,
                     adapt_to_max_length: bool = False, **kwargs):
        """
        Make a conversation tokens. Adapt interface of mindie-llm.

        Args:
            rank (int):
                The rank id.
            conversation (List[Dict[str, str]]):
                A conversation object or list of dicts.
            add_generation_prompt (bool, *optional*):
                Whether to end the prompt with the token(s) that indicate the start of an assistant message.
            adapt_to_max_length (bool, *optional*):
                Where input tokens should less max_length.

        Returns:
             context_tokens
        """
        self.rank = rank
        self.adapt_to_max_length = adapt_to_max_length
        context_tokens = self._apply_chat_template(conversation, add_generation_prompt=add_generation_prompt,
                                                   **kwargs)
        return context_tokens

    def _apply_chat_template(self, conversation: List[Dict[str, str]], **kwargs):
        """
        Converts a Conversation to a list of token ids.

        Args:
            conversation (List[Dict[str, str]]):
                A conversation object or list of dicts.

        Returns:
             input_ids
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("The tokenizer dose not implement apply_chat_template function.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")
        input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
        return input_ids


def _load_distributed_safetensors(model, strategy_path, load_safetensors):
    """Load distributed safetensors"""
    ms.load_distributed_checkpoint(
        network=model,
        predict_strategy=strategy_path,
        unified_safetensors_dir=load_safetensors,
        format='safetensors'
    )


def _load_safetensors(model, load_safetensors):
    """Load single safetensors"""
    sf_list = [sf for sf in os.listdir(load_safetensors) if sf.endswith('.safetensors')]
    if not sf_list:
        raise FileNotFoundError(f"There are no safetensors files under the given path {load_safetensors}.")
    for sf in sf_list:
        ms.load_checkpoint(
            ckpt_file_name=os.path.join(load_safetensors, sf),
            net=model,
            format='safetensors'
        )


def _check_valid_safetensors_path(path):
    """Check whether the safetensors path is valid"""
    if not isinstance(path, str) or isinstance(path, os.PathLike):
        raise ValueError(f"path must be a str, but got {path} as type {type(path)}.")
    if not os.path.exists(path):
        raise ValueError(f"path does not exist.")
    if contains_safetensors_files(path):
        return
    raise ValueError(f"load_checkpoint is not a valid path for safetensors.")
