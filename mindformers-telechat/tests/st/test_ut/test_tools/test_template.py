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
"""test Config"""
import inspect
import pytest
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig, \
    TransformerSwapConfig
from mindformers.modules.transformer import MoEConfig as MoEConfig_cls
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.register.template import CONFIG_NAME_TO_CLASS, CallbackConfig, ConfigTemplate, ContextConfig, \
    EvalCallbackConfig, EvalDatasetConfig, EvalDatasetTaskConfig, GeneralConfig, LrScheduleConfig, MetricConfig, \
    MoEConfig, ModelConfig, MsParallelConfig, OptimizerConfig, ParallelConfig, ProcessorConfig, RecomputeConfig, \
    RunnerConfig, TrainDatasetConfig, TrainDatasetTaskConfig, TrainerConfig, WrapperConfig, SwapConfig


def get_class_init_params_and_defaults(cls):
    """
    Get the initialization parameters and their default values for a given class.

    Args:
        cls: The class to introspect.

    Returns:
        A dictionary where keys are parameter names and values are their default values.
        If a parameter does not have a default value, it will be represented as `None`.
    """
    signature = inspect.signature(cls.__init__)
    params = signature.parameters

    result = {}
    for name, param in params.items():
        if name == "self":
            continue
        result[name] = param.default if param.default is not inspect.Parameter.empty else None

    return result

class TestGeneralConfig:
    """test general_config"""
    def setup_method(self):
        self.correct_input = {"run_mode": "train", "seed": 2}
        self.unexpected_input = {"run_mode": "train", "aaa": 1}

    def test_none_input(self):
        config = GeneralConfig.apply(None)
        for key in GeneralConfig.keys():
            assert key in config
            assert config[key] == getattr(GeneralConfig, key)
        for key in config.keys():
            assert key in GeneralConfig.keys()

    def test_empty_dict_input(self):
        config = GeneralConfig.apply({})
        for key in GeneralConfig.keys():
            assert key in config
            assert config[key] == getattr(GeneralConfig, key)
        for key in config.keys():
            assert key in GeneralConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            GeneralConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = GeneralConfig.apply(self.correct_input)
        for key in GeneralConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(GeneralConfig, key)
        for key in config.keys():
            assert key in GeneralConfig.keys()


class TestParallelConfig:
    """test parallel_config"""
    def setup_method(self):
        self.correct_input = {"data_parallel": 4, "model_parallel": 2}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = ParallelConfig.apply(None)
        config2 = ParallelConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in ParallelConfig.keys():
                assert key in config
                assert config[key] == getattr(ParallelConfig, key)
            for key in config.keys():
                assert key in ParallelConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            ParallelConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = ParallelConfig.apply(self.correct_input)
        for key in ParallelConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(ParallelConfig, key)
        for key in config.keys():
            assert key in ParallelConfig.keys()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_attribute_equal_to_class(self):
        """Test if the number of attributes is equal to the class attribute"""
        cls_init_params = get_class_init_params_and_defaults(TransformerOpParallelConfig)
        cls_init_params.pop("recompute")
        cls_init_params.pop("swap")
        assert len(cls_init_params) == len(ParallelConfig.keys())
        for key in ParallelConfig.keys():
            assert getattr(ParallelConfig, key) == cls_init_params[key]


class TestRecomputeConfig:
    """test recompute_config"""
    def setup_method(self):
        self.correct_input = {"recompute": True, "select_recompute": True}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = RecomputeConfig.apply(None)
        config2 = RecomputeConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in RecomputeConfig.keys():
                assert key in config
                assert config[key] == getattr(RecomputeConfig, key)
            for key in config.keys():
                assert key in RecomputeConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            RecomputeConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = RecomputeConfig.apply(self.correct_input)
        for key in RecomputeConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(RecomputeConfig, key)
        for key in config.keys():
            assert key in RecomputeConfig.keys()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_attribute_equal_to_class(self):
        cls_init_params = get_class_init_params_and_defaults(TransformerRecomputeConfig)
        assert len(cls_init_params) == len(RecomputeConfig.keys())
        for key in RecomputeConfig.keys():
            assert getattr(RecomputeConfig, key) == cls_init_params[key]


class TestSwapConfig:
    """test swap_config"""
    def setup_method(self):
        self.correct_input = {"swap": True}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = SwapConfig.apply(None)
        config2 = SwapConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in SwapConfig.keys():
                assert key in config
                assert config[key] == getattr(SwapConfig, key)
            for key in config.keys():
                assert key in SwapConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            SwapConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = SwapConfig.apply(self.correct_input)
        for key in SwapConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(SwapConfig, key)
        for key in config.keys():
            assert key in SwapConfig.keys()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_attribute_equal_to_class(self):
        cls_init_params = get_class_init_params_and_defaults(TransformerSwapConfig)
        assert len(cls_init_params) == len(SwapConfig.keys())
        for key in SwapConfig.keys():
            assert getattr(SwapConfig, key) == cls_init_params[key]


class TestMoEConfig:
    """test moe_config"""
    def setup_method(self):
        self.correct_input = {"expert_num": 4, "capacity_factor": 1.2}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = MoEConfig.apply(None)
        config2 = MoEConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in MoEConfig.keys():
                assert key in config
                assert config[key] == getattr(MoEConfig, key)
            for key in config.keys():
                assert key in MoEConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            MoEConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = MoEConfig.apply(self.correct_input)
        for key in MoEConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(MoEConfig, key)
        for key in config.keys():
            assert key in MoEConfig.keys()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_attribute_equal_to_class(self):
        cls_init_params = get_class_init_params_and_defaults(MoEConfig_cls)
        assert len(cls_init_params) == len(MoEConfig.keys())
        for key in MoEConfig.keys():
            assert getattr(MoEConfig, key) == cls_init_params[key]


class TestRunnerConfig:
    """test runner_config"""
    def setup_method(self):
        self.correct_input = {"batch_size": 4, "epochs": 2}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = RunnerConfig.apply(None)
        config2 = RunnerConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in RunnerConfig.keys():
                assert key in config
                assert config[key] == getattr(RunnerConfig, key)
            for key in config.keys():
                assert key in RunnerConfig.keys()

    def test_unexpected_input(self):
        with pytest.raises(KeyError, match="unexpected"):
            RunnerConfig.apply(self.unexpected_input)

    def test_correct_input(self):
        config = RunnerConfig.apply(self.correct_input)
        for key in RunnerConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(RunnerConfig, key)
        for key in config.keys():
            assert key in RunnerConfig.keys()


class TestMsParallelConfig:
    """test parallel"""
    def setup_method(self):
        self.correct_input = {"parallel_mode": 0, "full_batch": False, "aaa": 1}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = MsParallelConfig.apply(None)
        config2 = MsParallelConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in MsParallelConfig.keys():
                assert key in config
                assert config[key] == getattr(MsParallelConfig, key)
            for key in config.keys():
                assert key in MsParallelConfig.keys()

    def test_unexpected_input(self):
        config = MsParallelConfig.apply(self.unexpected_input)
        for key in MsParallelConfig.keys():
            assert key in config
            assert config[key] == getattr(MsParallelConfig, key)
        assert config["aaa"] == 1

    def test_correct_input(self):
        config = MsParallelConfig.apply(self.correct_input)
        for key in MsParallelConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(MsParallelConfig, key)
        assert config["aaa"] == 1


class TestContextConfig:
    """test context"""
    def setup_method(self):
        self.correct_input = {"mode": 1, "device_target": "CPU", "aaa": 1}
        self.unexpected_input = {"aaa": 1}

    def test_none_input(self):
        config1 = ContextConfig.apply(None)
        config2 = ContextConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in ContextConfig.keys():
                assert key in config
                assert config[key] == getattr(ContextConfig, key)
            for key in config.keys():
                assert key in ContextConfig.keys()

    def test_unexpected_input(self):
        config = ContextConfig.apply(self.unexpected_input)
        for key in ContextConfig.keys():
            assert key in config
            assert config[key] == getattr(ContextConfig, key)
        assert config["aaa"] == 1

    def test_correct_input(self):
        config = ContextConfig.apply(self.correct_input)
        for key in ContextConfig.keys():
            assert key in config
            if key in self.correct_input.keys():
                assert config[key] == self.correct_input[key]
            else:
                assert config[key] == getattr(ContextConfig, key)
        assert config["aaa"] == 1


class TestTrainDatasetConfig:
    """test train_dataset"""
    def setup_method(self):
        self.input = {"aaa": 1}

    def test_none_input(self):
        config = TrainDatasetConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = TrainDatasetConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_input(self):
        config = TrainDatasetConfig.apply(self.input)
        assert len(config) == 1
        assert config["aaa"] == 1


class TestTrainDatasetTaskConfig:
    """test train_dataset_task"""
    def setup_method(self):
        self.missing_required_input = {"aaa": 1}
        self.input = {"type": "class", "aaa": 1}

    def test_none_input(self):
        config = TrainDatasetTaskConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = TrainDatasetTaskConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            TrainDatasetTaskConfig.apply(self.missing_required_input)

    def test_input(self):
        config = TrainDatasetTaskConfig.apply(self.input)
        assert len(config) == 2
        assert config["type"] == "class"
        assert config["aaa"] == 1


class TestProcessorConfig:
    """test processor"""
    def setup_method(self):
        self.input_1 = {"aaa": 1}
        self.input_2 = {"type": "class", "aaa": 1}

    def test_none_input(self):
        config = ProcessorConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = ProcessorConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_input_1(self):
        config = ProcessorConfig.apply(self.input_1)
        assert len(config) == 1
        assert config["aaa"] == 1

    def test_input_2(self):
        config = ProcessorConfig.apply(self.input_2)
        assert len(config) == 2
        assert config["type"] == "class"
        assert config["aaa"] == 1


class TestEvalDatasetConfig:
    """test eval_dataset"""
    def setup_method(self):
        self.input = {"aaa": 1}

    def test_none_input(self):
        config = EvalDatasetConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = EvalDatasetConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_input(self):
        config = EvalDatasetConfig.apply(self.input)
        assert len(config) == 1
        assert config["aaa"] == 1


class TestEvalDatasetTaskConfig:
    """test eval_dataset_task"""
    def setup_method(self):
        self.missing_required_input = {"aaa": 1}
        self.input = {"type": "class", "aaa": 1}

    def test_none_input(self):
        config = EvalDatasetTaskConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = EvalDatasetTaskConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            EvalDatasetTaskConfig.apply(self.missing_required_input)

    def test_input(self):
        config = EvalDatasetTaskConfig.apply(self.input)
        assert len(config) == 2
        assert config["type"] == "class"
        assert config["aaa"] == 1


class TestTrainerConfig:
    """test trainer"""
    def setup_method(self):
        self.missing_required_input = {"aaa": 1}
        self.input = {"type": "class", "aaa": 1}

    def test_none_input(self):
        config = TrainerConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = TrainerConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            TrainerConfig.apply(self.missing_required_input)

    def test_input(self):
        config = TrainerConfig.apply(self.input)
        assert len(config) == 2
        assert config["type"] == "class"
        assert config["aaa"] == 1


class TestModelConfig:
    """test model_config"""
    def setup_method(self):
        self.missing_required_input = {"aaa": 1}
        self.input = {"model_config": "model_config", "arch": "arch", "aaa": 1}

    def test_none_input(self):
        config = ModelConfig.apply(None)
        assert isinstance(config, dict)
        assert not config
        config = ModelConfig.apply({})
        assert isinstance(config, dict)
        assert not config

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            ModelConfig.apply(self.missing_required_input)

    def test_input(self):
        config = ModelConfig.apply(self.input)
        assert len(config) == 3
        assert config["model_config"] == "model_config"
        assert config["arch"] == "arch"
        assert config["aaa"] == 1


class TestWrapperConfig:
    """test runner_wrapper"""
    def setup_method(self):
        self.correct_input = {"type": "type", "aaa": 1}
        self.missing_required_input = {"aaa": 1}

    def test_none_input(self):
        config1 = WrapperConfig.apply(None)
        config2 = WrapperConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in WrapperConfig.keys():
                assert key in config
                assert config[key] == getattr(WrapperConfig, key)
            for key in config.keys():
                assert key in WrapperConfig.keys()

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            ModelConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = WrapperConfig.apply(self.correct_input)
        assert len(config) == 2
        assert config["type"] == "type"
        assert config["aaa"] == 1


class TestOptimizerConfig:
    """test optimizer"""
    def setup_method(self):
        self.correct_input = {"type": "type", "aaa": 1}
        self.missing_required_input = {"aaa": 1}

    def test_none_input(self):
        config1 = OptimizerConfig.apply(None)
        config2 = OptimizerConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in OptimizerConfig.keys():
                assert key in config
                assert config[key] == getattr(OptimizerConfig, key)
            for key in config.keys():
                assert key in OptimizerConfig.keys()

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            OptimizerConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = OptimizerConfig.apply(self.correct_input)
        assert len(config) == 2
        assert config["type"] == "type"
        assert config["aaa"] == 1


class TestLrScheduleConfig:
    """test lr_schedule"""
    def setup_method(self):
        self.correct_input = {"type": "type", "aaa": 1}
        self.missing_required_input = {"aaa": 1}

    def test_none_input(self):
        config1 = LrScheduleConfig.apply(None)
        config2 = LrScheduleConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in LrScheduleConfig.keys():
                assert key in config
                assert config[key] == getattr(LrScheduleConfig, key)
            for key in config.keys():
                assert key in LrScheduleConfig.keys()

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            LrScheduleConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = LrScheduleConfig.apply(self.correct_input)
        assert len(config) == 2
        assert config["type"] == "type"
        assert config["aaa"] == 1


class TestMetricConfig:
    """test metric"""
    def setup_method(self):
        self.correct_input = {"type": "type", "aaa": 1}
        self.missing_required_input = {"aaa": 1}

    def test_none_input(self):
        config1 = MetricConfig.apply(None)
        config2 = MetricConfig.apply({})
        configs = [config1, config2]
        for config in configs:
            for key in MetricConfig.keys():
                assert key in config
                assert config[key] == getattr(MetricConfig, key)
            for key in config.keys():
                assert key in MetricConfig.keys()

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            MetricConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = MetricConfig.apply(self.correct_input)
        assert len(config) == 2
        assert config["type"] == "type"
        assert config["aaa"] == 1


class TestCallbackConfig:
    """test callbacks"""
    def setup_method(self):
        self.correct_input = [{"type": "type", "aaa": 1}, {"type": "MFLossMonitor", "aaa": 1}]
        self.missing_required_input = [{"aaa": 1}]

    def test_none_input(self):
        config1 = CallbackConfig.apply(None)
        config2 = CallbackConfig.apply([])
        configs = [config1, config2]
        for config in configs:
            assert len(config) == 2
            assert config[0]["type"] in ("MFLossMonitor", "ObsMonitor")
            assert config[1]["type"] in ("MFLossMonitor", "ObsMonitor")

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            CallbackConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = CallbackConfig.apply(self.correct_input)
        for callback in config:
            if callback["type"] == "MFLossMonitor":
                assert callback["aaa"] == 1

        assert len(config) == 3
        assert config[2]["type"] == "type"
        assert config[2]["aaa"] == 1


class TestEvalCallbackConfig:
    """test eval_callbacks"""
    def setup_method(self):
        self.correct_input = [{"type": "type", "aaa": 1}, {"type": "MFLossMonitor", "aaa": 1}]
        self.missing_required_input = [{"aaa": 1}]

    def test_none_input(self):
        config1 = EvalCallbackConfig.apply(None)
        config2 = EvalCallbackConfig.apply([])
        configs = [config1, config2]
        for config in configs:
            assert len(config) == 1
            assert config[0]["type"] == "ObsMonitor"

    def test_missing_required_input(self):
        with pytest.raises(KeyError, match="required"):
            EvalCallbackConfig.apply(self.missing_required_input)

    def test_correct_input(self):
        config = EvalCallbackConfig.apply(self.correct_input)
        assert len(config) == 3
        assert config[0]["type"] == "ObsMonitor"
        assert config[1]["type"] == "type"
        assert config[1]["aaa"] == 1
        assert config[2]["type"] == "MFLossMonitor"
        assert config[2]["aaa"] == 1


TRAIN_DEFAULT_CONFIGS = [
    "runner_wrapper",
    "optimizer",
    "lr_schedule",
    "recompute_config",
    "metric"
]

DEFAULT_CONFIGS = [
    "parallel_config",
    "parallel",
    "runner_config",
    "context",
    "moe_config"
]

callbacks = [
    "callbacks",
    "eval_callbacks"
]


def compare_default_callback(config, callback_name):
    cls = CONFIG_NAME_TO_CLASS[callback_name]
    # pylint: disable=W0212
    callback_list = cls._default_value()
    type_list = [callback["type"] for callback in callback_list]
    for callback in config[callback_name]:
        assert callback["type"] in type_list


def compare_default_config(config, target_configs):
    for sub_config_name in target_configs:
        sub_config = config[sub_config_name]
        cls = CONFIG_NAME_TO_CLASS[sub_config_name]
        for key in cls.keys():
            assert getattr(cls, key) == sub_config[key]


class TestTemplate:
    """test ConfigTemplate"""
    def setup_method(self):
        """initialize MindformerConfig"""
        self.train_config = MindFormerConfig(
            run_mode="train",
            trainer={"type": 1},
            model={"model_config": 1, "arch": 1},
            train_dataset={"a": 1},
            train_dataset_task={"type": 1}
        )
        self.train_eval_config = MindFormerConfig(
            run_mode="train",
            do_eval=True,
            trainer={"type": 1},
            model={"model_config": 1, "arch": 1},
            train_dataset={"a": 1},
            train_dataset_task={"type": 1},
            eval_dataset={"a": 1},
            eval_dataset_task={"type": 1}
        )
        self.predict_config = MindFormerConfig(
            run_mode="predict",
            trainer={"type": 1},
            model={"model_config": 1, "arch": 1},
            processor={"type": 1}
        )
        self.eval_config = MindFormerConfig(
            run_mode="eval",
            trainer={"type": 1},
            model={"model_config": 1, "arch": 1},
            eval_dataset={"a": 1},
            eval_dataset_task={"type": 1}
        )

    def test_none_input(self):
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
            ConfigTemplate.apply_template(None)

    def test_empty_dict_input(self):
        ConfigTemplate.apply_template({})

    def test_correct_train_config(self):
        """test which input is correct for train"""
        config = self.train_config
        ConfigTemplate.apply_template(config)
        compare_default_config(config, DEFAULT_CONFIGS)
        compare_default_config(config, TRAIN_DEFAULT_CONFIGS)
        compare_default_callback(config, "callbacks")

        assert config['trainer']["type"] == 1
        assert config['model']['model_config'] == 1
        assert config['model']['arch'] == 1
        assert config['train_dataset']['a'] == 1
        assert config['train_dataset_task']['type'] == 1

    def test_correct_train_eval_config(self):
        """test which input is correct for eval while training"""
        config = self.train_eval_config
        ConfigTemplate.apply_template(config)
        compare_default_config(config, DEFAULT_CONFIGS)
        compare_default_config(config, TRAIN_DEFAULT_CONFIGS)
        compare_default_callback(config, "callbacks")
        compare_default_callback(config, "eval_callbacks")

        assert config['trainer']["type"] == 1
        assert config['model']["model_config"] == 1
        assert config['model']["arch"] == 1
        assert config['train_dataset']["a"] == 1
        assert config['train_dataset_task']["type"] == 1
        assert config['eval_dataset']["a"] == 1
        assert config['eval_dataset_task']["type"] == 1

    def test_correct_predict_config(self):
        config = self.predict_config
        ConfigTemplate.apply_template(config)
        compare_default_config(config, DEFAULT_CONFIGS)

        assert config['trainer']["type"] == 1
        assert config['model']["model_config"] == 1
        assert config['model']["arch"] == 1
        assert config['processor']["type"] == 1

    def test_correct_eval_config(self):
        """test which input is correct for eval"""
        config = self.eval_config
        ConfigTemplate.apply_template(config)
        compare_default_config(config, DEFAULT_CONFIGS)
        compare_default_callback(config, "eval_callbacks")

        assert config['trainer']["type"] == 1
        assert config['model']["model_config"] == 1
        assert config['model']["arch"] == 1
        assert config['eval_dataset']["a"] == 1
        assert config['eval_dataset_task']["type"] == 1

    def test_overwrite_config(self):
        """test overwrite default value"""
        self.train_config["seed"] = 2024
        self.train_config["parallel_config"] = {"data_parallel": 2}
        self.train_config["runner_wrapper"] = {"type": 1}
        config = self.train_config
        ConfigTemplate.apply_template(config)

        assert config['seed'] == 2024
        assert config['parallel_config']["data_parallel"] == 2
        assert config['parallel_config']["model_parallel"] == 1
        assert config['runner_wrapper']["type"] == 1

    def test_wrong_run_mode_1(self):
        config = MindFormerConfig()
        ConfigTemplate.apply_template(config)
        config = MindFormerConfig(run_mode="xxx")
        ConfigTemplate.apply_template(config)

    def test_trainer_missing_key(self):
        config = MindFormerConfig(run_mode="train",
                                  trainer={"a": 1})
        with pytest.raises(KeyError, match="missing a required key: type"):
            ConfigTemplate.apply_template(config)

    def test_model_missing_key_1(self):
        config = MindFormerConfig(run_mode="train",
                                  trainer={"type": 1},
                                  model={"model_config": 1})
        with pytest.raises(KeyError, match="missing a required key: arch"):
            ConfigTemplate.apply_template(config)

    def test_model_missing_key_2(self):
        config = MindFormerConfig(run_mode="train",
                                  trainer={"type": 1},
                                  model={"arch": 1})
        with pytest.raises(KeyError, match="missing a required key: model_config"):
            ConfigTemplate.apply_template(config)

    def test_unexpected_key(self):
        self.train_config["a"] = 1
        config = self.train_config
        ConfigTemplate.apply_template(config)
        assert "a" in config.keys()
