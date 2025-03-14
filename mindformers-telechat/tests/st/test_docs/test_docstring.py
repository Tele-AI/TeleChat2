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
Test docstring of external APIs.
How to run this:
pytest tests/st/test_docs/test_docstring.py
"""
import importlib
import inspect
import re
from typing import Callable

import pytest

from mindspore import nn


def dynamic_import(full_path: str) -> type:
    """Dynamically import a class or method."""
    try:
        module_name, obj_name = full_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        obj = getattr(module, obj_name)
        return obj
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {full_path}: \"{e}\"."
                          f"If its import path is modified, modify the interface_map in this test case. ")


def has_custom_init(cls: type) -> bool:
    """check whether the class has a custom __init__"""
    return '__init__' in cls.__dict__


def get_method_signature_of_cls(cls: type, method_name: str) -> inspect.Signature:
    """Gets the full signature of the class methods, including those inherited from the parent class."""
    # Get the method object of the class
    method = getattr(cls, method_name, None)
    if method is None:
        raise ValueError(f"The class {cls} does not have a method names {method_name}.")

    # Iterate through the MRO (method resolution order) to find method definitions
    for base_cls in cls.__mro__:
        if method_name in base_cls.__dict__:
            func = base_cls.__dict__[method_name]
            # If the method is a class method or a static method, it needs to get its original function
            if isinstance(func, (classmethod, staticmethod)):
                return inspect.signature(func.__func__)
            return inspect.signature(func)

    raise ValueError(f"Method {method_name} not found in MRO (method resolution order) of class {cls.__name__}.")


def parse_args_in_docstring(api_name: str, docstring: str) -> dict:
    """Parses a Google-style docstring to extract parameter definitions in the Args section."""
    args_pattern = re.compile(r"Args:\s*\n(.*?)(?=\n\s*(Inputs|Returns)|$)", re.DOTALL)
    args_section = args_pattern.search(docstring)

    if not args_section:
        return {}

    param_pattern = re.compile(r"^\s*\*{0,2}(\w+)\s\(([\w\s.,\[\]()`]+)\):\s(.*?)(?=\n\s*\w+\s*\(|$)", re.MULTILINE)
    matches = param_pattern.findall(args_section.group(1))

    if not matches:
        raise ValueError(f"No formatted parameter information was found "
                         f"in the Args section of the interface {api_name}.")

    params = {}
    for param_name, param_type, _ in matches:
        is_optional = "optional" in param_type.lower()
        params[param_name] = {
            "type": param_type.replace("optional", "").strip(','),
            "required": not is_optional,
        }

    return params


def parse_inputs_in_docstring(api_name: str, docstring: str) -> dict:
    """Parses a Google-style docstring to extract parameter definitions in the Inputs section."""
    inputs_pattern = re.compile(r"Inputs:\s*\n(.*?)(?=\n\s*Outputs|$)", re.DOTALL)
    inputs_section = inputs_pattern.search(docstring)

    if not inputs_section:
        raise ValueError(f"The docstring of interface {api_name} does not define the Inputs section.")

    param_pattern = re.compile(
        r"^\s*-\s\*\*(?:\\\*)?(\w+)\*\*\s\(([\w\s.,\[\]()`]+)\)\s-\s(.*?)(?=\n\s*-\s*\*\*\w+\*\*\s*\(|$)",
        re.MULTILINE)
    matches = param_pattern.findall(inputs_section.group(1))

    if not matches:
        raise ValueError(f"No formatted parameter information was found "
                         f"in the Inputs section of the interface {api_name}.")

    params = {}
    for param_name, param_type, _ in matches:
        is_optional = "optional" in param_type.lower()
        params[param_name] = {
            "type": param_type.replace("optional", "").strip(','),
            "required": not is_optional,
        }

    return params


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestDocstring:
    """A test class for testing docstring."""

    def setup_class(self):
        """Init interface map."""
        self.interface_map = {
            "mindformers.AutoConfig": ["from_pretrained", "register", "show_support_list"],
            "mindformers.AutoModel": ["from_config", "from_pretrained", "register"],
            "mindformers.AutoModelForCausalLM": ["from_config", "from_pretrained", "register"],
            "mindformers.AutoProcessor": ["from_pretrained", "register"],
            "mindformers.AutoTokenizer": ["from_pretrained", "register"],
            "mindformers.Trainer": ["evaluate", "finetune", "predict", "train"],
            "mindformers.TrainingArguments": ["convert_args_to_mindformers_config", "get_moe_config",
                                              "get_parallel_config", "get_recompute_config", "get_warmup_steps",
                                              "set_dataloader", "set_logging", "set_lr_scheduler", "set_optimizer",
                                              "set_save", "set_training"],
            "mindformers.pipeline": None,
            "mindformers.run_check": None,
            "mindformers.ModelRunner": None,
            "mindformers.core.build_context": None,
            "mindformers.core.get_context": None,
            "mindformers.core.init_context": None,
            "mindformers.core.set_context": None,
            "mindformers.core.CrossEntropyLoss": None,
            "mindformers.core.AdamW": None,
            "mindformers.core.Came": None,
            "mindformers.core.LearningRateWiseLayer": None,
            "mindformers.core.ConstantWarmUpLR": None,
            "mindformers.core.LinearWithWarmUpLR": None,
            "mindformers.core.CosineWithWarmUpLR": None,
            "mindformers.core.CosineWithRestartsAndWarmUpLR": None,
            "mindformers.core.PolynomialWithWarmUpLR": None,
            "mindformers.core.CosineAnnealingLR": None,
            "mindformers.core.CosineAnnealingWarmRestarts": None,
            "mindformers.core.CheckpointMonitor": None,
            "mindformers.core.EvalCallBack": None,
            "mindformers.core.MFLossMonitor": None,
            "mindformers.core.TrainingStateMonitor": None,
            "mindformers.core.ProfileMonitor": None,
            "mindformers.core.SummaryMonitor": None,
            "mindformers.core.EntityScore": ["clear", "eval", "update"],
            "mindformers.core.EmF1Metric": ["clear", "eval", "update"],
            "mindformers.core.PerplexityMetric": ["clear", "eval", "update"],
            "mindformers.core.PromptAccMetric": ["clear", "eval", "update"],
            "mindformers.dataset.CausalLanguageModelDataset": None,
            "mindformers.dataset.KeyWordGenDataset": None,
            "mindformers.dataset.MultiTurnDataset": None,
            "mindformers.generation.GenerationConfig": None,
            "mindformers.generation.GenerationMixin": ["chat", "forward", "generate", "infer", "postprocess"],
            "mindformers.models.PreTrainedModel": ["can_generate", "from_pretrained", "post_init",
                                                   "register_for_auto_class", "save_pretrained"],
            "mindformers.models.PretrainedConfig": ["from_dict", "from_json_file", "from_pretrained",
                                                    "get_config_dict", "save_pretrained", "to_dict",
                                                    "to_diff_dict", "to_json_file", "to_json_string"],
            "mindformers.models.PreTrainedTokenizer": ["convert_ids_to_tokens", "convert_tokens_to_ids",
                                                       "get_added_vocab", "num_special_tokens_to_add",
                                                       "prepare_for_tokenization", "tokenize"],
            "mindformers.models.PreTrainedTokenizerFast": ["convert_ids_to_tokens", "convert_tokens_to_ids",
                                                           "get_added_vocab", "num_special_tokens_to_add",
                                                           "set_truncation_and_padding", "train_new_from_iterator"],
            "mindformers.models.multi_modal.ModalContentTransformTemplate": ["batch", "build_conversation_input_text",
                                                                             "build_labels", "build_modal_context",
                                                                             "get_need_update_output_items",
                                                                             "post_process", "process_predict_query",
                                                                             "process_train_item"],
            "mindformers.models.LlamaForCausalLM": None,
            "mindformers.models.LlamaConfig": None,
            "mindformers.models.LlamaTokenizer": ["build_inputs_with_special_tokens",
                                                  "create_token_type_ids_from_sequences",
                                                  "get_special_tokens_mask"],
            "mindformers.models.LlamaTokenizerFast": ["build_inputs_with_special_tokens",
                                                      "save_vocabulary",
                                                      "update_post_processor"],
            "mindformers.models.ChatGLM2ForConditionalGeneration": None,
            "mindformers.models.ChatGLM2Config": None,
            "mindformers.models.ChatGLM3Tokenizer": None,
            "mindformers.models.ChatGLM4Tokenizer": None,
            "mindformers.modules.OpParallelConfig": None,
            "mindformers.pet.models.LoraModel": None,
            "mindformers.pet.pet_config.PetConfig": None,
            "mindformers.pet.pet_config.LoraConfig": None,
            "mindformers.pipeline.MultiModalToTextPipeline": None,
            "mindformers.tools.register.MindFormerModuleType": None,
            "mindformers.tools.register.MindFormerRegister": ["get_cls", "get_instance", "get_instance_from_cfg",
                                                              "is_exist", "register", "register_cls"],
            "mindformers.tools.MindFormerConfig": ["merge_from_dict"],
            "mindformers.wrapper.MFTrainOneStepCell": None,
            "mindformers.wrapper.MFPipelineWithLossScaleCell": None,
        }

    def setup_method(self):
        """Init errors list."""
        self.error_list = []

    def check_class_docstring(self, cls: type, full_path: str) -> None:
        """Checks that the arguments in the class docstring match the arguments in the class definition."""
        docstring = inspect.getdoc(cls)
        if not docstring:
            self.error_list.append(f"{full_path}: Missing docstring.")
            return

        # Parses the Args section of the class's docstring.
        try:
            doc_params = parse_args_in_docstring(full_path, docstring)
        except ValueError as e:
            self.error_list.append(str(e))
            return

        # Get __init__ method signature
        if has_custom_init(cls):
            target_method_name = "__init__"
        else:
            target_method_name = "__new__"

        try:
            sig = get_method_signature_of_cls(cls, target_method_name)
        except ValueError as e:
            self.error_list.append(str(e))
            return
        init_params = sig.parameters

        # Checking arguments consistency
        for name, param_doc in doc_params.items():
            if name not in init_params:
                self.error_list.append(f"{full_path}: Argument '{name}' in docstring is not defined in "
                                       f"{target_method_name} method.")

            elif param_doc["required"] and init_params[name].default is not inspect.Parameter.empty:
                self.error_list.append(f"{full_path}: The docstring specifies that the argument '{name}' should be "
                                       f"required, but there is a default value in the {target_method_name} method.")
        for name in init_params:
            if name not in doc_params and name != 'cls' and name != 'self' and name != 'args' and name != 'kwargs':
                self.error_list.append(f"{full_path}: There is a redundant argument '{name}' in the "
                                       f"{target_method_name} method that is not defined in the docstring or is "
                                       f"defined in an incorrect format.")

        # Whether the class is a subclass of mindspore.nn.Cell,
        # means that the arguments to its construct method need to be checked
        if not issubclass(cls, nn.Cell):
            return
        # The class does not define construct method
        if 'construct' not in cls.__dict__:
            return

        # Parses the Inputs section of the class's docstring.
        try:
            inputs_params = parse_inputs_in_docstring(cls.__name__, docstring)
        except ValueError as e:
            self.error_list.append(str(e))
            return

        # Get construct method signature
        construct_method = getattr(cls, 'construct', None)
        if not callable(construct_method):
            raise ValueError(f"{full_path}: The class {cls.__name__} does not define construct method.")

        sig = inspect.signature(construct_method)
        construct_params = sig.parameters

        # Checking arguments consistency
        for name, param_doc in inputs_params.items():
            if name not in construct_params:
                self.error_list.append(f"{full_path}: Argument '{name}' in docstring is not defined in construct "
                                       f"method.")
            elif param_doc["required"] and construct_params[name].default is not inspect.Parameter.empty:
                self.error_list.append(f"{full_path}: The docstring specifies that the argument '{name}' should be "
                                       f"required, but there is a default value in the function signature.")
        for name in construct_params:
            if name not in inputs_params and name != 'self' and name != 'kwargs':
                self.error_list.append(f"{full_path}: There is a redundant argument '{name}' in the construct "
                                       f"method that is not defined in the docstring or is defined in an incorrect "
                                       f"format.")

    def check_method_docstring(self, func: Callable, full_path: str, cls: type = None) -> None:
        """Checks that the arguments in the method docstring match the arguments in the method definition."""
        docstring = inspect.getdoc(func)
        if not docstring:
            self.error_list.append(f"{full_path}: Missing docstring.")
            return

        if cls:
            try:
                sig = get_method_signature_of_cls(cls, func.__name__)
            except ValueError as e:
                self.error_list.append(str(e))
                return
        else:
            sig = inspect.signature(func)
        func_params = sig.parameters

        try:
            doc_params = parse_args_in_docstring(full_path, docstring)
        except ValueError as e:
            self.error_list.append(str(e))
            return

        for name, param_doc in doc_params.items():
            if name not in func_params:
                self.error_list.append(f"{full_path}: Argument '{name}' in docstring is not defined in the function "
                                       f"signature.")
            elif param_doc["required"] and func_params[name].default is not inspect.Parameter.empty:
                self.error_list.append(f"{full_path}: The docstring specifies that the argument '{name}' should be "
                                       f"required, but there is a default value in the function signature.")
        for name in func_params:
            if name not in doc_params and name != 'cls' and name != 'self' and name != 'args' and name != 'kwargs':
                self.error_list.append(f"{full_path}: There is a redundant argument '{name}' in the function signature "
                                       f"that is not defined in the docstring or is defined in an incorrect format.")

    def test_docstring(self):
        """Test docstring for all the interfaces."""
        for class_or_func_path, methods in self.interface_map.items():
            obj = dynamic_import(class_or_func_path)

            if inspect.isclass(obj):
                self.check_class_docstring(obj, class_or_func_path)

                if not methods:
                    continue

                # check method
                for method_name in methods:
                    if not hasattr(obj, method_name):
                        raise AttributeError(f"The class {class_or_func_path} does not have method {method_name}. "
                                             f"If the method has been deleted, "
                                             f"modify the interface_map in this test case. ")
                    method = getattr(obj, method_name)
                    self.check_method_docstring(method, class_or_func_path + "." + method_name, obj)

            else:
                self.check_method_docstring(obj, class_or_func_path)

        if self.error_list:
            raise AssertionError(f"Checking arguments consistency failed! The inconsistencies are listing below:\n" +
                                 " ".join(f"{i + 1}. {s}" for i, s in enumerate(self.error_list)))
