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
"""test SwapConfig"""
import pytest
from mindformers.modules.transformer import TransformerSwapConfig


def get_correct_config():
    """Provide a error example of the swap config format."""
    config = {
        "layer_swap": [
            {"backward_prefetch": 2, "layers": True},
        ],
        "op_swap": {
            "mul": [{"backward_prefetch": 1, "layers": [2]}],
            "w1": [{"backward_prefetch": 2, "layers": True}],
        },
        "swap": True,
        "default_prefetch": 1
    }
    return config


def get_error_config():
    """Provide a correct example of the swap config format."""
    config = {
        "layer_swap": [
            {"backward_prefetch": 1, "layers": [2]},
            {"backward_prefetch": 2, "layers": True},
        ],
        "op_swap": [
            {"op_name": 'mul', "backward_prefetch": 1, "layers": [2]},
            {"op_name": 'w1', "backward_prefetch": 2, "layers": True},
        ],
        "swap": True,
        "default_prefetch": 1
    }
    return config


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_swap_invalid_input():
    """
    Feature: Activation offload adapts mindformers
    Description: Test whether the swap input is invalid
    Expectation: Run successfully
    """
    config = get_error_config()
    values = (3, 3.0, [3.0], [[3.0]], [True], (3.0,), ((3.0,),))
    for v in values:
        with pytest.raises(ValueError) as exc_info:
            config["layer_swap"][1]["layers"] = v
            TransformerSwapConfig(**config)
            assert "Invalid layer_swap configuration: 3. Expected 'layers' to be a list, tuple, or bool." in str(
                exc_info.value)

    config["layer_swap"][1]["layers"] = [3]
    for v in values:
        with pytest.raises(ValueError) as exc_info:
            config["op_swap"][1]["layers"] = v
            TransformerSwapConfig(**config)
            assert "Invalid op_swap: w1 configuration: 3. Expected 'layers' to be a list, tuple, or bool." in str(
                exc_info.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_swap_conflicts():
    """
    Feature: Activation offload adapts mindformers
    Description: Test whether confilcts exist in the swap config
    Expectation: Run successfully
    """
    config = get_error_config()
    with pytest.raises(ValueError) as exc_info:
        TransformerSwapConfig(**config)
        assert "Invalid layer_swap configuration at index 1" in str(exc_info.value)
    config["layer_swap"][0]["backward_prefetch"] = 2
    swap_config = TransformerSwapConfig(**config)
    assert swap_config.to_dict() == get_correct_config()

    config2 = get_error_config()
    config2["layer_swap"][0]["backward_prefetch"] = 2
    with pytest.raises(ValueError) as exc_info:
        config2["op_swap"].append({"op_name": 'mul', "backward_prefetch": 2, "layers": True})
        TransformerSwapConfig(**config2)
        assert "Invalid op_swap: mul configuration at index 1:" in str(exc_info.value)
    config2["op_swap"][0]["backward_prefetch"] = 2
    swap_config2 = TransformerSwapConfig(**config2)
    expected = get_correct_config()
    expected["op_swap"]["mul"] = [{'backward_prefetch': 2, 'layers': True}]
    assert swap_config2.to_dict() == expected


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_swap_normal():
    """
    Feature: Activation offload adapts mindformers
    Description: Test whether the format of swap config is correct
    Expectation: Run successfully
    """
    config = get_error_config()
    config["layer_swap"][0]["backward_prefetch"] = 2
    swap_config = TransformerSwapConfig(**config)
    assert swap_config.to_dict() == get_correct_config()

    valid_values = ([3], [[3]], [3, 5], [[3, 4]], (3,), (3, 4), (3, 4), ((3, 4),))
    config1 = get_error_config()
    for v in valid_values:
        config1["layer_swap"][1]["layers"] = v
        swap_config1 = TransformerSwapConfig(**config1)
        expected = get_correct_config()
        expected["layer_swap"] = [{'backward_prefetch': 1, 'layers': [2]}, {'backward_prefetch': 2, 'layers': v}]
        assert swap_config1.to_dict() == expected

    config2 = get_error_config()
    config2["layer_swap"][0]["backward_prefetch"] = 2
    for v in valid_values:
        config2["op_swap"][1]["layers"] = v
        swap_config2 = TransformerSwapConfig(**config2)
        expected = get_correct_config()
        expected["op_swap"]['w1'] = [{"backward_prefetch": 2, "layers": v}]
        assert swap_config2.to_dict() == expected
