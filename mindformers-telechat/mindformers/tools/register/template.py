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
"""Config and Template"""
import copy

from mindformers.tools import logger


class Config:
    """
    A base class for applying structured configuration.

    This class serves as a blueprint for handling configuration data by providing methods
    to validate, update, and manage key-value pairs from config files or dictionaries. It
    supports configuration inputs as dictionaries, allowing for flexible configuration
    management in workflows.

    Attributes:
        _name (str): The name of the configuration. This can be set by subclasses to
                     differentiate between different types of configurations.
        _raise_error_for_unexpected_key (bool): If True, raises an error when encountering
                     unexpected keys in the input configuration. If False, ignores them.
        _support_none_input (bool): If True, allows the configuration input to be None.
                     If False, raises a ValueError when the input is empty.
        _required_keys (list): A list of keys that must be present in the configuration.
                     If any required key is missing, a KeyError is raised.

    Note:
        This is a base class and is often subclassed to implement specific configurations
        with additional attributes and required keys. The behavior of how unexpected keys
        and empty configurations are handled can be customized by setting class attributes
        in subclasses.
    """
    _name = ""
    _raise_error_for_unexpected_key = True
    _support_none_input = True
    _required_keys = []

    @classmethod
    def apply(cls, config):
        """
        Apply the configuration dictionary to the class, with validation.

        Args:
            config (Union[dict, None]): The configuration to apply. It can be a dictionary.
                If None is provided, the behavior depends on `_support_none_input`.

        Returns:
            dict: A dictionary containing the final configuration with default and updated values.

        Raises:
            TypeError: If the input config is neither a dict.
            ValueError: If the config is empty when `_support_none_input` is False.
            KeyError: If a required key is missing or an unexpected key is found in the config.
        """
        if config is None:
            config = {}

        if not isinstance(config, dict):
            raise TypeError(f"The input config should be a dict, but get {type(config)}")

        if not config and not cls._support_none_input:
            raise ValueError(f"The config '{cls._name}' is empty. Please check the yaml file.")

        if not config:
            return cls._none_process()
        result = cls._initialize_result()
        return cls._update_value(result, config)

    @classmethod
    def _none_process(cls):
        logger.warning(f"The input config {cls._name} is empty.")
        return cls._default_value()

    @classmethod
    def _initialize_result(cls):
        return cls._default_value()

    @classmethod
    def _update_value(cls, result, config):
        """
        Update the result dictionary with values from the input config.

        Args:
            result (dict): The initial result dictionary to be updated.
            config (dict): The input configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.

        Raises:
            KeyError: If required keys are missing from any configuration dictionary.
        """
        config = copy.deepcopy(config)
        for required_key in cls._required_keys:
            if required_key not in config:
                raise KeyError(f"The config '{cls._name}' is missing a required key: {required_key}.")
            result[required_key] = config.pop(required_key)

        for key, value in config.items():
            if key in cls.keys() or not cls._raise_error_for_unexpected_key:
                result[key] = value
            else:
                raise KeyError(f"The config '{cls._name}' gets an unexpected key: {key}")
        return result

    @classmethod
    def keys(cls):
        """
        Get all non-private attribute names of the class.

        Returns:
            list: A list of attribute names defined in the class (excluding private ones).
        """
        return [k for k in cls.__dict__ if not k.startswith("_")]

    @classmethod
    def _default_value(cls):
        return {key: getattr(cls, key) for key in cls.keys()}

    @classmethod
    @property
    def name(cls):
        """
        Get the name of the configuration.

        Returns:
            str: The name of the configuration class.
        """
        return cls._name


class SpecConfig(Config):
    _raise_error_for_unexpected_key = False

    @classmethod
    def _initialize_result(cls):
        return {}


class ListConfig(Config):
    """
    A configuration class for handling list-based configurations.

    This class extends the `Config` base class to handle configurations provided as lists.
    It supports updating configurations based on input lists and allows for specifying default values.

    It is especially useful for cases where configurations are structured as a list of dictionaries,
    each representing a configuration block.

    Attributes:
        _name (str): The name identifier for the configuration.
        _raise_error_for_unexpected_key (bool): If False, unexpected keys in the config will not raise errors.
        _support_none_input (bool): Determines if None input is supported without raising errors.
        _required_keys (list): A list of keys that are required in each configuration dictionary.
    """
    _raise_error_for_unexpected_key = False

    @classmethod
    def apply(cls, config):
        """
        Apply the list-based configuration to the class, with validation.

        Args:
            config (Union[list, None]): The configuration to apply, expected to be a list of dictionaries.
                If None is provided, the behavior depends on `_support_none_input`.

        Returns:
            list: A list containing the final configuration with default and updated values.

        Raises:
            TypeError: If the input config is not a list.
            ValueError: If the config is empty when `_support_none_input` is False.
        """
        if config is None:
            config = []

        if not isinstance(config, list):
            raise TypeError(f"The input config should be a list, but get {type(config)}")

        if not config and not cls._support_none_input:
            raise ValueError(f"The config '{cls._name}' is empty. Please check the yaml file.")

        if not config:
            return cls._none_process()
        result = cls._initialize_result()
        return cls._update_value_list(result, config)

    @classmethod
    def _update_value_list(cls, result, config):
        """
        Update the list-based result with values from the input config list.

        Args:
            result (list): The initial result list to be updated.
            config (list): The input configuration list, where each element is expected to be a dictionary.

        Returns:
            list: The updated list containing configurations.
        """
        config = copy.deepcopy(config)
        for value in config:
            res_tmp = {}
            res_tmp = cls._update_value(res_tmp, value)
            if value["type"] in cls._types_to_index():
                result[cls._types_to_index()[value["type"]]] = res_tmp.copy()
            else:
                result.append(res_tmp.copy())
        return result

    @classmethod
    def _default_value(cls):
        return [getattr(cls, key) for key in cls.keys()]

    @classmethod
    def _types_to_index(cls):
        dic = {}
        for i, value in enumerate(cls._default_value()):
            dic[value["type"]] = i
        return dic


class GeneralConfig(Config):
    """general configs in yaml"""
    seed = 1
    output_dir = "./output"
    run_mode = None
    use_parallel = False
    resume_training = False

    # checkpoint
    load_checkpoint = ""
    load_ckpt_format = "ckpt"
    auto_trans_ckpt = False
    transform_process_num = 1
    src_strategy_path_or_dir = ""
    only_save_strategy = False
    load_ckpt_async = False

    # eval while training
    do_eval = False
    eval_step_interval = 100
    eval_epoch_interval = -1

    # skip dataset
    ignore_data_skip = False
    data_skip_steps = None

    # profile
    profile = False
    profile_communication = False
    profile_memory = True
    init_start_profile = False
    profile_start_step = 1
    profile_stop_step = 10
    profile_rank_ids = None
    profile_pipeline = False
    profile_level = 1

    layer_scale = False
    layer_decay = 0.65
    lr_scale = False
    lr_scale_factor = 256
    micro_batch_interleave_num = 1
    remote_save_url = None
    save_file = None

    # predict
    input_data = None
    predict_batch_size = None
    adapter_id = None

    # mf context
    exclude_cann_cpu = False
    train_precision_sync = False
    infer_precision_sync = False
    postprocess_use_numpy = False

    _name = "general_config"


class ParallelConfig(Config):
    """model parallel config"""
    data_parallel = 1
    model_parallel = 1
    context_parallel = 1
    expert_parallel = 1
    pipeline_stage = 1
    micro_batch_num = 1
    seq_split_num = 1
    use_seq_parallel = False
    optimizer_shard = None
    gradient_aggregation_group = 4
    vocab_emb_dp = True
    context_parallel_algo = "colossalai_cp"
    ulysses_degree_in_cp = 1
    mem_coeff = 0.1

    _name = "parallel_config"


class RecomputeConfig(Config):
    recompute = False
    select_recompute = False
    parallel_optimizer_comm_recompute = False
    select_comm_recompute = False
    mp_comm_recompute = True
    recompute_slice_activation = False

    _name = "recompute_config"


class MoEConfig(Config):
    """moe parallel config"""
    expert_num = 1
    capacity_factor = 1.1
    aux_loss_factor = 0.05
    num_experts_chosen = 1
    expert_group_size = None
    group_wise_a2a = False
    comp_comm_parallel = False
    comp_comm_parallel_degree = 2
    save_token_distribution = False
    cur_layer = 0
    enable_cold_hot_expert = False
    update_step = 10000
    hot_expert_num = 0
    cold_token_percent = 1.0
    moe_module_name = ""
    routing_policy = "TopkRouterV1"
    norm_topk_prob = True
    enable_sdrop = False
    use_fused_ops_topkrouter = False
    router_dense_type = "float32"
    shared_expert_num = 0
    use_shared_expert_gating = False
    max_router_load = 128 * 1024
    topk_method = "greedy"
    topk_group = None
    n_group = None
    first_k_dense_replace = True
    moe_intermediate_size = 1407
    routed_scaling_factor = 1.0
    aux_loss_types = None
    aux_loss_factors = None
    z_loss_factor = 0.
    balance_via_topk_bias = False
    topk_bias_update_rate = 0.
    use_allgather_dispatcher = False
    moe_shared_expert_overlap = False
    expert_model_parallel = None
    use_gating_sigmoid = False

    _name = "moe_config"


class RunnerConfig(Config):
    batch_size = 1
    epochs = 1
    sink_mode = 1
    sink_size = 1
    gradient_accumulation_steps = 1
    num_classes = 1
    stop_step = 0

    _name = "runner_config"


class MsParallelConfig(Config):
    """parallel config for mindspore.set_auto_parallel_context()"""
    parallel_mode = 1
    full_batch = True
    search_mode = "sharding_propagation"
    enable_parallel_optimizer = False
    gradients_mean = False
    enable_alltoall = False
    strategy_ckpt_save_file = "./ckpt_strategy.ckpt"

    _raise_error_for_unexpected_key = False
    _name = "parallel"


class ContextConfig(Config):
    mode = 0
    device_target = "Ascend"
    device_id = 0
    max_device_memory = "58GB"
    max_call_depth = 10000

    _raise_error_for_unexpected_key = False
    _name = "context"


class TrainDatasetConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "train_dataset"


class TrainDatasetTaskConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "train_dataset_task"
    _required_keys = ["type"]


class ProcessorConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "processor"


class EvalDatasetConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "eval_dataset"


class EvalDatasetTaskConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "eval_dataset_task"
    _required_keys = ["type"]


class TrainerConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "trainer"
    _required_keys = ["type"]


class ModelConfig(Config):
    _raise_error_for_unexpected_key = False

    _name = "model"
    _required_keys = ["model_config", "arch"]


class WrapperConfig(SpecConfig):
    _name = "runner_wrapper"
    _required_keys = ["type"]

    type = "MFTrainOneStepCell"
    use_clip_grad = True
    scale_sense = {
        "type": "DynamicLossScaleUpdateCell",
        "loss_scale_value": 65536,
        "scale_factor": 2,
        "scale_window": 1000
    }


class OptimizerConfig(SpecConfig):
    _name = "optimizer"
    _required_keys = ["type"]

    type = "AdamW"
    betas = [0.9, 0.999]
    learning_rate = 5.e-5
    eps = 1.e-8
    weight_decay = 0.0


class LrScheduleConfig(SpecConfig):
    _name = "lr_schedule"
    _required_keys = ["type"]

    type = "CosineWithWarmUpLR"
    learning_rate = 5.e-5
    lr_end = 0.
    warmup_ratio = 0.
    total_steps = -1


class MetricConfig(SpecConfig):
    _name = "metric"
    _required_keys = ["type"]

    type = "PerplexityMetric"


class CallbackConfig(ListConfig):
    callback1 = {"type": "MFLossMonitor"}
    callback2 = {"type": "ObsMonitor"}

    _name = "callbacks"
    _required_keys = ["type"]


class EvalCallbackConfig(ListConfig):
    callback1 = {"type": "ObsMonitor"}

    _name = "eval_callbacks"
    _required_keys = ["type"]


CONFIG_NAME_TO_CLASS = {
    "general_config": GeneralConfig,
    "parallel_config": ParallelConfig,
    "recompute_config": RecomputeConfig,
    "moe_config": MoEConfig,
    "runner_config": RunnerConfig,
    "parallel": MsParallelConfig,
    "context": ContextConfig,
    "train_dataset": TrainDatasetConfig,
    "train_dataset_task": TrainDatasetTaskConfig,
    "processor": ProcessorConfig,
    "eval_dataset": EvalDatasetConfig,
    "eval_dataset_task": EvalDatasetTaskConfig,
    "trainer": TrainerConfig,
    "model": ModelConfig,
    "runner_wrapper": WrapperConfig,
    "optimizer": OptimizerConfig,
    "lr_schedule": LrScheduleConfig,
    "metric": MetricConfig,
    "callbacks": CallbackConfig,
    "eval_callbacks": EvalCallbackConfig,
}


class ConfigTemplate:
    """
    A template handler for managing and applying configurations.

    This class organizes and applies configuration templates based on different run modes
    such as 'train', 'eval', 'predict', and 'finetune'. It categorizes configurations into
    general, training, evaluation, and prediction sections to structure and streamline the
    application of various configurations for workflows.

    Attributes:
        general_configs (list): A list of general configuration sections.
        train_configs (list): A list of configuration sections specific to training.
        do_eval_configs (list): A list of configuration sections used during evaluation in training.
        predict_configs (list): A list of configuration sections for prediction tasks.
        eval_configs (list): A list of evaluation configuration sections.
        _run_modes (list): Supported modes of operation ('train', 'eval', 'predict', 'finetune').
    """
    general_configs = [
        "general_config",
        "runner_config",
        "context",
        "parallel",
        "trainer",
        "model",
        "moe_config",
        "parallel_config"
    ]

    train_configs = [
        "recompute_config",
        "runner_wrapper",
        "optimizer",
        "lr_schedule",
        "metric",
        "train_dataset",
        "train_dataset_task",
        "callbacks"
    ]

    do_eval_configs = [
        "eval_dataset",
        "eval_dataset_task",
        "eval_callbacks"
    ]

    predict_configs = [
        "processor"
    ]

    eval_configs = [
        "eval_dataset",
        "eval_dataset_task",
        "eval_callbacks",
        "metric"
    ]

    _run_modes = ['train', 'eval', 'predict', 'finetune']

    @classmethod
    def apply_template(cls, config):
        """
        Apply the appropriate configuration template based on the run mode.

        Args:
            config (dict): The configuration dict containing the run mode
                and other relevant settings.

        Returns:
            dict: A new dict with the applied template.
        """
        run_mode = config.get('run_mode', None)
        if run_mode not in cls._run_modes:
            logger.warning(f"The specified run_mode '{run_mode}' is invalid. Expected one of {cls._run_modes}.")
            template = cls.general_configs
        elif run_mode in ['train', 'finetune']:
            template = cls._train_template(config.get("do_eval", False))
        elif run_mode == "predict":
            template = cls._predict_template()
        else:
            template = cls._eval_template()
        cls._apply_template(config, template, run_mode)


    @classmethod
    def _apply_template(cls, config, template, run_mode):
        """
        Apply a specific template to the configuration.

        Args:
            config (dict): The original configuration dict.
            template (list): A list of configuration sections to be applied.

        Returns:
            dict: A new configuration dict with sections applied from the template.
        """
        cls._aggregate_general_config(config)

        new_config = {}
        for sub_config in template:
            class_ = CONFIG_NAME_TO_CLASS[sub_config]
            new_config[sub_config] = class_.apply(config.pop(sub_config, None))

        unused_config = [key for key in config.keys()]
        if unused_config:
            logger.warning(f"Some configs in yaml are useless for {run_mode}: {unused_config}")
        config.update(new_config)

        cls._scatter_general_config(config)

    @classmethod
    def _train_template(cls, do_eval):
        template = []
        template.extend(cls.general_configs)
        template.extend(cls.train_configs)
        if do_eval:
            template.extend(cls.do_eval_configs)
        return template

    @classmethod
    def _predict_template(cls):
        template = []
        template.extend(cls.general_configs)
        template.extend(cls.predict_configs)
        return template

    @classmethod
    def _eval_template(cls):
        template = []
        template.extend(cls.general_configs)
        template.extend(cls.eval_configs)
        return template

    @staticmethod
    def _aggregate_general_config(config):
        """
        Aggregate all general configuration keys into a single 'general_config' section.

        Args:
            config (dict): The original configuration dict.

        Returns:
            dict: The configuration dict with aggregated general settings.
        """
        general_config = {}
        general_keys = []
        for key in CONFIG_NAME_TO_CLASS["general_config"].keys():
            if key in config:
                general_keys.append(key)
        for key in general_keys:
            general_config[key] = config.pop(key)
        config["general_config"] = general_config

    @staticmethod
    def _scatter_general_config(config):
        """
        Scatter the 'general_config' settings back into the main configuration.

        Args:
            config (dict): The configuration dict with 'general_config'.

        Returns:
            dict: The updated configuration dict with scattered general settings.
        """
        general_config = config.pop("general_config")
        for k, v in general_config.items():
            config[k] = v
        return config

