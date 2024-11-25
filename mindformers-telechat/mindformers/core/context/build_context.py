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
"""Build context."""

import os
import json
from typing import Union

import psutil

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers.tools import PARALLEL_MODE, MODE, get_output_subpath, check_in_modelarts
from mindformers.tools.check_rules import get_server_num
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import check_in_dynamic_cluster, set_strategy_save_path
from mindformers.trainer.config_args import ContextConfig, ParallelContextConfig
from mindformers.trainer.training_args import TrainingArguments
from mindformers.utils import get_cann_workqueue_cores

CONTEXT = {
    'mode': 'GRAPH_MODE',
    'device_target': 'Ascend',
    'device_id': 0,
    'save_graphs': False
}

MF_CONFIG = {
    'run_mode': None,
    'exclude_cann_cpu': False,
    'train_precision_sync': False,
    'infer_precision_sync': False,
    'postprocess_use_numpy': False
}

PARALLEL = {
    'parallel_mode': 'SEMI_AUTO_PARALLEL',
    'gradients_mean': True
}

CONTEXT_INSTANCE = None


class _Context:
    """
    _Context is the environment in which operations are executed

    Note:
        Create a context through instantiating Context object is not recommended.
        should use build_context() to instantiating the context since Context is a singleton.
    """
    _instance = None

    # pylint: disable=W0613
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        """Build context."""
        self.config = config

        self.set_train_precision_sync(self.config.get('train_precision_sync', None))
        self.set_infer_precision_sync(self.config.get('infer_precision_sync', None))

        local_rank, device_num = self.init_ms_context()
        config.device_num = device_num
        config.local_rank = local_rank
        use_cpu_affinity = os.environ.get("CPU_AFFINITY")
        if use_cpu_affinity and (use_cpu_affinity == '1' or use_cpu_affinity.lower() == 'true'):
            ds.config.set_numa_enable(True)
            self.cpu_affinity(local_rank, device_num)
            logger.info(f"cpu_affinity, rank_id: {local_rank}, device_num: {device_num}")

        speedup_config_json = os.environ.get("MS_ENABLE_NUMA")
        if speedup_config_json and os.path.exists(os.path.realpath(speedup_config_json)):
            speedup_real_path = os.path.realpath(speedup_config_json)
            with open(speedup_real_path, 'r') as f:
                speedup_config = json.load(f)
                self.cpu_affinity_by_config(local_rank, speedup_config)

        if config.parallel.get("strategy_ckpt_load_file"):
            self.set_ms_auto_parallel_context(strategy_ckpt_load_file=config.parallel.strategy_ckpt_load_file)

        if config.context.get('runtime_num_threads') is None and self.check_runtime_num_threads_version():
            self.set_ms_context(runtime_num_threads=1)
            logger.info("The current MindSpore version is %s,"
                        "and set the default runtime_num_threads to 1.", ms.__version__)

    def init_ms_context(self):
        """Context initialization for MindSpore.

        Args:
            config (MindFormerConfig): configurations

        Returns: rank_id, device_num.
        """
        if self.config.use_parallel:
            self.set_pipeline_stage()

        self.set_predict_context_config()

        self.set_check_context_config(self.config.context)
        self.set_check_parallel_config(self.config.parallel)

        device_num = 1
        rank_id = 0
        self.config.context['mode'] = MODE.get(self.config.context.get('mode'))

        self.set_ms_context(max_device_memory=self.config.context.get('max_device_memory'),
                            mode=self.config.context.get('mode'))

        if self.config.use_parallel:
            device_id = int(os.getenv('DEVICE_ID', '0'))  # 0 ~ 7
            self.config.context['device_id'] = device_id
            if check_in_dynamic_cluster():
                # for dynamic cluster, we should not set device id in context.
                self.config.context.pop('device_id', None)
            self.config.parallel['parallel_mode'] = PARALLEL_MODE.get(self.config.parallel.get('parallel_mode'))
            self.set_ms_context(**self.config.context)
            try:
                init()
            # pylint: disable=W0702
            except:
                raise RuntimeError("Notice: if you are trying to run with a single device, please set "
                                   "use_parallel=False. If not, please check the error message above.")
            rank_id = get_rank()  # local_rank
            device_num = get_group_size()  # world_size
            self.config.parallel.setdefault('device_num', device_num)
            context.reset_auto_parallel_context()
            set_strategy_save_path(self.config.parallel)
            self.set_ms_auto_parallel_context(**self.config.parallel)
        else:
            self.set_ms_context(**self.config.context)

        if context.get_auto_parallel_context("parallel_mode") == "auto_parallel":
            set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
        else:
            set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=True)
        _set_multi_subgraphs()

        return rank_id, device_num

    def set_run_mode(self, run_mode):
        if run_mode is not None:
            self.config.run_mode = run_mode

    def set_ms_context(self, **kwargs):
        context.set_context(**kwargs)

    def get_ms_context(self, attr_key):
        return context.get_context(attr_key)

    def set_ms_auto_parallel_context(self, **kwargs):
        context.set_auto_parallel_context(**kwargs)

    def check_runtime_num_threads_version(self):
        """check mindspore version that need to set the runtime_num_threads to 1"""
        return bool(ms.__version__ < "2.3.0")

    def set_predict_context_config(self):
        """Set predict context config."""
        run_mode = self.config.get('run_mode', None)
        if run_mode is not None and run_mode.strip() in ["predict", "eval"] and self.config.model.model_config.use_past:
            os.environ['MS_ALLOC_CONF'] = os.environ.setdefault('MS_ALLOC_CONF', 'enable_vmm:False')
            os.environ['RUN_MODE'] = run_mode
            jit_level = self.config.context.get("jit_level", "O0")
            infer_boost = self.config.context.get("infer_boost", "on")
            logger.info(f"Predict context config, jit_level: {jit_level}, infer_boost: {infer_boost}")

            if jit_level == "O1":
                raise ValueError("jit_level O1 is not support in predict mode.")

            if jit_level == "O2" and infer_boost == "on":
                raise ValueError("Only infer_boost set off can set jit_level to O2 in predict mode.")

            self.set_ms_context(jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    def set_pipeline_stage(self):
        """Set pipeline stage number."""
        input_stages = 1
        if self.config.parallel_config.pipeline_stage:
            input_stages = self.config.parallel_config.pipeline_stage
        if self.config.parallel.auto_pipeline:
            micro_batch = self.config.parallel_config.micro_batch_num
            servers = get_server_num()
            final_stages = max(input_stages, servers)
            final_stages = min(final_stages, micro_batch)
            logger.info(f"Automatic pipeline stage provider will search in [1...{final_stages}], "
                        f"where {final_stages} = min( max( stages input: {input_stages}, servers: {servers}), "
                        f"micro batch: {micro_batch})")
        else:
            final_stages = input_stages

        self.config.parallel_config.pipeline_stage = final_stages
        if final_stages > 1:
            self.config.parallel.pipeline_stages = final_stages

    def cpu_affinity_by_config(self, rank_id, speedup_config):
        """cpu affinity by config json"""
        group_core_size = speedup_config.get("each_process_cpu_core_size", 0)
        rank_info = "rank_" + str(rank_id)
        if group_core_size != 0:
            rank_info = "rank_0"
            group_core_size *= rank_id
        cur_bind_info = speedup_config.get(rank_info)
        if cur_bind_info is not None and cur_bind_info.get("python"):
            used_cpus = [i + group_core_size for i in cur_bind_info.get("python")]
            cur_process = psutil.Process()
            cur_process.cpu_affinity(used_cpus)
            logger.warning(f"custom cpu affinity policy, rank_id: {rank_id}, cpus: {used_cpus}")

    def cpu_affinity(self, rank_id, rank_size):
        """cpu affinity"""
        count = psutil.cpu_count()
        current_process = psutil.Process()
        used_cpus_num = count // rank_size
        used_cpus = [i for i in range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)]
        cann_used_cpus = get_cann_workqueue_cores(rank_id)
        logger.info(f"cann workqueue will use following cpus: {cann_used_cpus}")
        used_cpus = list(set(used_cpus) - set(cann_used_cpus))
        if not used_cpus:
            # cann setup all cpus, disable the binding cores
            logger.warning(f"cann use cpus({cann_used_cpus}), model get empty cpu list, disable binding cores")
            used_cpus = [i for i in range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)]
        current_process.cpu_affinity(used_cpus)

    def set_check_context_config(self, config):
        """Set context config."""
        mode = config.get('mode')
        if mode is None:
            config.setdefault('mode', 0)
        if mode not in MODE.keys():
            raise IndexError(f'Running mode should be in {MODE.keys()}, but get {mode}')

        device = config.get('device_id')
        if device is None:
            config.setdefault('device_id', 0)

        max_device_memory = config.get('max_device_memory')
        if max_device_memory is None:
            config.setdefault('max_device_memory', '1024GB')

        save_graph = config.get('save_graphs')
        if save_graph:
            save_graphs_path = config.get('save_graphs_path') if not check_in_modelarts() else None
            if save_graphs_path is None:
                save_graphs_path = get_output_subpath("debug/graphs_info", append_rank=False)
            config['save_graphs_path'] = save_graphs_path
        enable_dump = config.get('enable_dump')
        if enable_dump:
            save_dump_path = config.get('save_dump_path') if not check_in_modelarts() else None
            if save_dump_path is None:
                save_dump_path = get_output_subpath("debug/dump_info", append_rank=False)
            config['save_dump_path'] = save_dump_path

    def set_check_parallel_config(self, config):
        """Set parallel config."""
        parallel_mode = config.get('parallel_mode')
        if parallel_mode is None:
            config.setdefault('parallel_mode', 0)

        if PARALLEL_MODE.get(config.get('parallel_mode')) not in \
                [context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL] and \
                config.get('full_batch'):
            logger.info("full_batch will be forced to False when the parallel mode is stand_alone or data_parallel")
            config.setdefault('full_batch', False)

        if parallel_mode not in PARALLEL_MODE.keys():
            raise IndexError(f'Running parallel mode should be in {PARALLEL_MODE.keys()}, but get {parallel_mode}')

    def set_train_precision_sync(self, switch):
        """Control training precision synchronization"""
        if switch is None:
            return

        self.config.train_precision_sync = switch
        if switch:
            os.environ['HCCL_DETERMINISTIC'] = 'true'
            os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
            os.environ['TE_PARALLEL_COMPILER'] = '1'
            self.set_ms_context(deterministic="ON")
        else:
            os.environ['HCCL_DETERMINISTIC'] = 'false'
            os.environ['ASCEND_LAUNCH_BLOCKING'] = '0'
            os.environ['TE_PARALLEL_COMPILER'] = '0'
            self.set_ms_context(deterministic="OFF")

    def set_infer_precision_sync(self, switch):
        """Control infer precision synchronization"""
        if switch is None:
            return

        self.config.infer_precision_sync = switch
        if switch:
            os.environ['CUSTOM_MATMUL_SHUFFLE'] = 'off'
            os.environ['LCCL_DETERMINISTIC'] = '1'
        else:
            os.environ['CUSTOM_MATMUL_SHUFFLE'] = 'on'
            os.environ['LCCL_DETERMINISTIC'] = '0'


def _context(config=None):
    """
    Get the global _context, if context is not created, create a new one.

    Returns:
        _Context, the global context in PyNative mode.
    """
    global CONTEXT_INSTANCE
    if CONTEXT_INSTANCE is None:
        CONTEXT_INSTANCE = _Context(config)
    return CONTEXT_INSTANCE


def init_context(use_parallel=False, context_config=None, parallel_config=None):
    """
    Initialize the context.

    Args:
        use_parallel (bool): Whether to use parallel, default: False.
        context_config (Union[dict, ContextConfig]): The context config, default: None.
        parallel_config (Union[dict, ParallelContextConfig]): The parallel context config, default: None.

    Returns:
        - Int, the local_rank number.
        - Int, the total available devices number.

    Examples:
        >>> from mindformers import init_context
        >>> init_context(use_parallel=False)
    """
    if context_config is None:
        context_config = CONTEXT
    elif isinstance(context_config, ContextConfig):
        context_config = context_config.__dict__

    if parallel_config is None:
        parallel_config = PARALLEL
    elif isinstance(parallel_config, ParallelContextConfig):
        parallel_config = parallel_config.__dict__

    config = {
        'use_parallel': use_parallel,
        'context': context_config,
        'parallel': parallel_config
    }
    ctx = build_context(config)
    return ctx.config.local_rank, ctx.config.device_num


def build_context(config: Union[dict, MindFormerConfig, TrainingArguments]):
    """
    Build the context from config.

    Note:
        parameter config must contain keys: 'context', 'parallel', when config is dict.

    Args:
        config (Union[dict, MindFormerConfig, TrainingArguments]): The configuration to initialize the context.
            This can be a dictionary, a MindFormerConfig instance, or a TrainingArguments instance.

    Returns:
        _Context, The instantiated context.

    Examples:
        >>> from mindformers import build_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
    """
    if isinstance(config, TrainingArguments):
        config = config.convert_args_to_mindformers_config()

    config['context'] = MindFormerConfig(**{**CONTEXT, **config['context']})
    config['parallel'] = MindFormerConfig(**{**PARALLEL, **config['parallel']})
    config['parallel_config'] = config.parallel_config if config.get('parallel_config', None) else MindFormerConfig()
    if isinstance(config, MindFormerConfig):
        for k, v in MF_CONFIG.items():
            if k not in config:
                config[k] = v
    else:
        config = MindFormerConfig(**{**MF_CONFIG, **config})

    ctx = _context(config)
    return ctx


def set_context(run_mode=None, **kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program. If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        Currently only run_mode belongs to MindFormers context. The kwargs will be passed to MindSpore set_context.

    Args:
        run_mode (str): The mode of the model behaviour. Must be in ['train', 'finetune', 'eval', 'predict'].
        **kwargs: MindSpore context arguments.

    Examples:
        >>> from mindformers import build_context, set_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> set_context(max_device_memory='59GB')
    """
    ctx = _context()
    ctx.set_run_mode(run_mode)
    ctx.set_train_precision_sync(kwargs.pop('train_precision_sync', None))
    ctx.set_infer_precision_sync(kwargs.pop('infer_precision_sync', None))

    for k in list(kwargs.keys()):
        if k in MF_CONFIG:
            v = kwargs.pop(k)
            ctx.config[k] = v

    ctx.set_ms_context(**kwargs)
    for k, v in kwargs.items():
        tmp = ctx.config.context.get(k, None)
        ctx.config.context[k] = v
        if context.get_context(k) != v:
            if tmp:
                ctx.config.context[k] = tmp
            else:
                del ctx.config.context[k]


def get_context(attr_key):
    """
    Get context attribute value according to the input key.
    If some attributes are not set, they will be automatically obtained.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> from mindformers import build_context, get_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> get_context('max_device_memory')
    """
    ctx = _context()
    if attr_key in MF_CONFIG:
        return ctx.config.get(attr_key, None)
    val = ctx.get_ms_context(attr_key)
    ctx.config.context['attr_key'] = val
    return val
