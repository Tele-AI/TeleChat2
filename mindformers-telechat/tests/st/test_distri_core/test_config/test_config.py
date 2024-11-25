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
"""test configs and initializations"""

import pytest

from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    init_configs_from_dict,
    BaseConfig,
    TrainingConfig,
    ModelParallelConfig,
    OptimizerConfig,
    DatasetConfig,
    LoraConfig,
    TransformerConfig
)

class AConfig(BaseConfig):
    config_name = "a_config"
    def __init__(self, a, b, c, **kwargs):
        self.a = a
        self.b = b
        self.c = c

        self.update_attrs(**kwargs)

@AConfig.validator("a")
# pylint: disable=W0613
def validate_a(config_instance, value):
    if value < 0:
        raise ValueError("a must be non-negative")
    return value

class BConfig(BaseConfig):
    config_name = "b_config"
    def __init__(self, d, e, f, a_config, **kwargs):
        self.d = d
        self.e = e
        self.f = f
        self.a_config = a_config

        self.update_attrs(**kwargs)

BConfig.register_depended_config(AConfig)

class CConfig(BaseConfig):
    config_name = "c_config"
    def __init__(self, g, h, i, a_config=None, **kwargs):
        self.g = g
        self.h = h
        self.i = i
        self.a_config = a_config

        self.update_attrs(**kwargs)

CConfig.register_depended_config(AConfig, optional=True)


class DConfig(BaseConfig):
    config_name = 'd_config'
    def __init__(self, j, k, l, **kwargs):
        self.j = j
        self.k = k
        self.l = l

        self.update_attrs(**kwargs)


class EConfig(BaseConfig):
    config_name = 'e_config'
    def __init__(self, m, f_config, **kwargs):
        self.m = m
        self.f_config = f_config

        self.update_attrs(**kwargs)


class FConfig(BaseConfig):
    config_name = 'f_config'
    def __init__(self, n, e_config, **kwargs):
        self.n = n
        self.e_config = e_config

        self.update_attrs(**kwargs)

EConfig.register_depended_config(FConfig)
FConfig.register_depended_config(EConfig)



@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_base():
    """
    Feature: init configs from yaml, the basic test.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./base.yaml"
    all_config = init_configs_from_yaml(config_path)
    assert all_config.a_config.a == 1
    assert all_config.a_config.b == 2
    assert all_config.a_config.c == 3

    assert all_config.b_config.d == 4
    assert all_config.b_config.e == 5
    assert all_config.b_config.f == 6


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_dict():
    """
    Feature: init configs from dict
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    raw_dict = {
        "a_config": {"a": 1, "b": 2, "c": 3},
        "b_config": {"d": 4, "e": 5, "f": 6},
    }
    all_config = init_configs_from_dict(raw_dict)
    assert all_config.a_config.a == raw_dict["a_config"]["a"]
    assert all_config.a_config.b == raw_dict["a_config"]["b"]
    assert all_config.a_config.c == raw_dict["a_config"]["c"]

    assert all_config.b_config.d == raw_dict["b_config"]["d"]
    assert all_config.b_config.e == raw_dict["b_config"]["e"]
    assert all_config.b_config.f == raw_dict["b_config"]["f"]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_config_list():
    """
    Feature: init configs with config list
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./base.yaml"
    a_config, b_config = init_configs_from_yaml(config_path, [AConfig, BConfig])
    assert a_config.a == 1
    assert a_config.b == 2
    assert a_config.c == 3

    assert b_config.d == 4
    assert b_config.e == 5
    assert b_config.f == 6


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_missing_config():
    """
    Feature: init configs with missing config
    Description: Test to initialize configs from yaml with full config list
    Expectation: raise ValueError
    """
    config_path = "./missing.yaml"
    try:
        _ = init_configs_from_yaml(config_path)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_illegal_value():
    """
    Feature: init configs with illegal value
    Description: Test to initialize configs from yaml with full config list
    Expectation: raise ValueError
    """
    config_path = "./illegal.yaml"
    try:
        _ = init_configs_from_yaml(config_path)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_dict_value():
    """
    Feature: init configs from yaml, with dict value.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./dict_value.yaml"
    all_config = init_configs_from_yaml(config_path)
    assert all_config.a_config.a == 1
    assert all_config.a_config.b == 2
    assert all_config.a_config.c.m == 3

    assert all_config.b_config.d == 4
    assert all_config.b_config.e == 5
    assert all_config.b_config.f == 6


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_optional_config():
    """
    Feature: init configs from yaml, with optional config.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./optional.yaml"
    all_config = init_configs_from_yaml(config_path)
    assert all_config.c_config.a_config is None

    assert all_config.c_config.g == 4
    assert all_config.c_config.h == 5
    assert all_config.c_config.i == 6

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_no_depended_config():
    """
    Feature: init configs from yaml without depended config.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./no_depended.yaml"
    all_config = init_configs_from_yaml(config_path)
    assert all_config.a_config.a == 1
    assert all_config.a_config.b == 2
    assert all_config.a_config.c == 3

    assert all_config.d_config.j == 4
    assert all_config.d_config.k == 5
    assert all_config.d_config.l == 6

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_no_validation_func():
    """
    Feature: init configs from yaml without validation func.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./no_validation_func.yaml"
    all_config = init_configs_from_yaml(config_path)

    assert all_config.d_config.j == 4
    assert all_config.d_config.k == 5
    assert all_config.d_config.l == 6


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_kwargs():
    """
    Feature: init configs from yaml with extra kwargs.
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./kwargs.yaml"
    all_config = init_configs_from_yaml(config_path)
    assert all_config.a_config.a == 1
    assert all_config.a_config.b == 2
    assert all_config.a_config.c == 3
    assert all_config.a_config.z == 10

    assert all_config.b_config.d == 4
    assert all_config.b_config.e == 5
    assert all_config.b_config.f == 6


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_cycle_dependency():
    """
    Feature: init configs from yaml, with cycle dependency.
    Description: Test to initialize configs from yaml with full config list
    Expectation: raise ValueError
    """
    config_path = "./cycle_dependency.yaml"
    try:
        _ = init_configs_from_yaml(config_path)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_init_configs_from_yaml_with_full_config_list():
    """
    Feature: init configs from yaml with full config list
    Description: Test to initialize configs from yaml with full config list
    Expectation: success
    """
    config_path = "./full.yaml"

    _ = init_configs_from_yaml(
        config_path,
        [TrainingConfig, ModelParallelConfig, OptimizerConfig, DatasetConfig, LoraConfig, TransformerConfig],
    )

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_init_configs_from_yaml_with_partial_config_list():
    """
    Feature: init configs from yaml with partial config list
    Description: Test to initialize configs from yaml with partial config list
    Expectation: success
    """
    config_path = "./full.yaml"

    _ = init_configs_from_yaml(
        config_path, [TrainingConfig]
    )

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_init_configs_from_yaml_without_config_list():
    """
    Feature: init configs from yaml without config list
    Description: Test to initialize configs from yaml without config list
    Expectation: success
    """
    config_path = "./partial.yaml"

    all_config = init_configs_from_yaml(config_path)
    assert hasattr(all_config, "extra_config")
