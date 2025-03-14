# Copyright 2022 Huawei Technologies Co., Ltd
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
pipeline
"""
import importlib
import os
import traceback
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from mindspore import Model, set_context

from mindformers.models.auto import AutoConfig, AutoModel, AutoTokenizer, AutoImageProcessor
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import (BaseAudioProcessor, BaseImageProcessor,
                                PreTrainedModel, PreTrainedTokenizerBase,
                                build_processor, build_tokenizer, build_network)
from mindformers.models.auto import TOKENIZER_MAPPING, IMAGE_PROCESSOR_MAPPING
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import CONFIG_NAME
from mindformers.tools.logger import logger
from mindformers.tools.hub.dynamic_module_utils import \
    get_class_from_dynamic_module
from mindformers.tools.hub.hub import cached_file, extract_commit_hash
from mindformers.tools.register import MindFormerConfig

from .build_pipeline import build_pipeline
from .registry_constant import (MULTI_MODEL_CONFIGS,
                                NO_FEATURE_EXTRACTOR_TASKS,
                                NO_IMAGE_PROCESSOR_TASKS, NO_TOKENIZER_TASKS,
                                PIPELINE_REGISTRY)

SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
# reversed constant for feature extractor
FEATURE_EXTRACTOR_MAPPING = OrderedDict()


class Backend(Enum):
    MS = "ms"


def pipeline(
        task: str = None,
        model: Optional[Union[str, PreTrainedModel, Model, Tuple[str, str]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_processor: Optional[BaseImageProcessor] = None,
        audio_processor: Optional[BaseAudioProcessor] = None,
        backend: Optional[str] = "ms",
        **kwargs: Any):
    r"""Pipeline for executing the inference flow of integrated tasks and models.

    Args:
        task (str, optional):
            The task, now support ['text_generation', 'image_to_text_generation',
            'multi_modal_to_text_generation']. Default: ``None``.
        model (Union[str, PreTrainedModel, Model, Tuple[str, str]], optional):
            The model used for task. Default: ``None``.
        tokenizer (PreTrainedTokenizerBase, optional):
            The tokenizer of the model. Default: ``None``.
        image_processor (BaseImageProcessor, optional):
            The image processor of the model. Default: ``None``.
        audio_processor (BaseAudioProcessor, optional):
            The audio processor of the model. Default: ``None``.
        backend (str, optional):
            The inference backend. Currently, only "ms" is supported. Default: ``"ms"``.
        **kwargs (Any):
            Refers to the kwargs description of the corresponding task pipeline.

    Returns:
        A task pipeline.

    Raises:
        KeyError: If the task or model is not supported.

    Examples:
        >>> from mindformers import build_context
        >>> from mindformers import AutoModel, AutoTokenizer, pipeline
        >>> # Construct inputs
        >>> inputs = ["I love Beijing, because", "LLaMA is a", "Huawei is a company that"]
        >>> # Initialize the environment
        >>> build_context({
        ...     'context': {'mode': 0, 'jit_config': {'jit_level': 'O0', 'infer_boost': 'on'}},
        ...     'parallel': {},
        ...     'parallel_config': {}})
        >>> # Tokenizer instantiation
        >>> tokenizer = AutoTokenizer.from_pretrained('llama2_7b')
        >>> # Model instantiation
        >>> # Download the weights of the corresponding model from the HuggingFace model library,
        >>> # Refer to the README.md of the model to convert the weights to ckpt format.
        >>> model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path="path/to/llama2_7b.ckpt",
        ...                                   use_past=True)
        >>> # The pipeline performs inference task.
        >>> text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
        >>> outputs = text_generation_pipeline(inputs, max_length=512, do_sample=False)
        >>> for output in outputs:
        ...     print(output)
        'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I ......]
        'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and ......]
        'text_generation_text': [Huawei is a company that has been around for a long time. ......]
    """

    if backend == Backend.MS.value:
        if is_experimental_mode(model, **kwargs):
            audio_processor = kwargs.pop("feature_extractor", audio_processor)
            task_pipeline = get_ms_experimental_pipeline(task, model,
                                                         tokenizer=tokenizer,
                                                         feature_extractor=audio_processor,
                                                         image_processor=image_processor,
                                                         **kwargs)
        else:
            task_pipeline = get_ms_pipeline(task, model, tokenizer, image_processor, audio_processor, **kwargs)
    else:
        raise ValueError(f"The inference backend \"{backend}\" is not supported,"
                         f"please select a backend from [\"ms\"]")
    return task_pipeline


def get_ms_pipeline(task, model, tokenizer, image_processor, audio_processor, **kwargs):
    """get mindspore infer pipeline."""
    if task not in SUPPORT_PIPELINES.keys():
        raise KeyError(f"{task} is not supported by pipeline. please select"
                       f" a task from {SUPPORT_PIPELINES.keys()}.")
    if isinstance(model, str):
        support_model_name = SUPPORT_PIPELINES[task].keys()
        if model not in support_model_name:
            raise KeyError(
                f"model must be in {support_model_name} when model's type is string, but get {model}.")
        model_name = model
        model = None
    else:
        model_name = "common"
    pipeline_config = MindFormerConfig(SUPPORT_PIPELINES.get(task).get(model_name))

    if model is None:
        batch_size = kwargs.get("batch_size", None)
        build_names = ["batch_size", "use_past", "seq_length"]
        build_args = {}
        for build_name in build_names:
            if build_name in kwargs:
                build_args[build_name] = kwargs.pop(build_name)
        model = build_network(pipeline_config.model, default_args=build_args)
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
    if image_processor is None and hasattr(pipeline_config.processor, 'image_processor'):
        image_processor = build_processor(pipeline_config.processor.image_processor)
    if audio_processor is None and hasattr(pipeline_config.processor, 'audio_processor'):
        audio_processor = build_processor(pipeline_config.processor.audio_processor)
    if tokenizer is None:
        tokenizer = build_tokenizer(pipeline_config.processor.tokenizer, tokenizer_name=model_name)
    task_pipeline = build_pipeline(class_name=task,
                                   model=model,
                                   image_processor=image_processor,
                                   audio_processor=audio_processor,
                                   tokenizer=tokenizer,
                                   **kwargs)
    return task_pipeline


def get_ms_experimental_pipeline(
        task: str = None,
        model: Optional[Union[str, PreTrainedModel, Model, Tuple[str, str]]] = None,
        config: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        feature_extractor: Optional[BaseAudioProcessor] = None,
        image_processor: Optional[BaseImageProcessor] = None,
        framework: str = "ms",
        revision: Optional[str] = None,
        use_fast: Optional[bool] = False,
        token: Optional[Union[str, bool]] = None,
        mode: Optional[int] = None,
        device_id: Optional[int] = None,
        device_target: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_class: Optional[Any] = None,
        **kwargs):
    r"""Pipeline for downstream tasks

    Args:
        task (str):
            The supported task could be selected from PIPELINE_REGISTRY.get_supported_tasks().
        model (Optional[Union[str, PreTrainedModel]]):
            The model used for task.
        tokenizer (Optional[PreTrainedTokenizerBase]):
            The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]):
            The image processor of the model.
        feature_extractor (Optional[BaseAudioProcessor]):
            The feature_ xtractor of the model. Preserved keyword for now.
        framework(str):
            The inference backend. Default "ms", now support ["ms", "mslite"].
        **kwargs:
            Refers to the kwargs description of the corresponding task pipeline.

    Returns:
        A task pipeline.

    Raises:
        KeyError: If the task or model is not supported.

    Examples:
        >>> from mindformers import pipeline
        >>> from mindformers.tools.image_tools import load_image
        >>> classifier = pipeline("zero_shot_image_classification",
            candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
        >>> img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
            "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> classifier(img)
        [[{'score': 0.99995565, 'label': 'sunflower'},
        {'score': 2.5318595e-05, 'label': 'toy'},
        {'score': 9.903885e-06, 'label': 'dog'},
        {'score': 6.75336e-06, 'label': 'tree'},
        {'score': 2.396818e-06, 'label': 'cat'}]]
    """
    if model_kwargs is None:
        model_kwargs = {}

    if feature_extractor is not None:
        logger.warning("feature_extractor is a reserved field and is currently not in use")

    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)

    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    # do not infer task or model, need to be specified both
    if task is None or model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without a task or a model being specified. "
            "Please provide a task class and a model."
        )
    if isinstance(model, Path):
        model = os.path.realpath(str(model))

    if framework == Backend.MS.value:
        logger.info("Initializing ms pipeline.")
    else:
        raise ValueError(f"The inference framework \"{framework}\" is not supported,"
                         f"please select a framework from [\"ms\"]")

    if commit_hash is None:
        pretrained_model_name_or_path = None
        if isinstance(config, str):
            pretrained_model_name_or_path = config
        elif config is None and isinstance(model, str):
            pretrained_model_name_or_path = model

        if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            hub_kwargs["_commit_hash"] = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            hub_kwargs["_commit_hash"] = getattr(config, "_commit_hash", None)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # pylint: disable=W0212
        hub_kwargs["_commit_hash"] = config._commit_hash
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # pylint: disable=W0212
        hub_kwargs["_commit_hash"] = config._commit_hash

    custom_tasks = {}
    # pylint: disable=C1801
    if config is not None and len(getattr(config, "custom_pipelines", {})) > 0:
        custom_tasks = config.custom_pipelines

    # Retrieve the task
    if task in custom_tasks:
        normalized_task = task
        targeted_task, _ = clean_custom_task(custom_tasks[task])
        if pipeline_class is None:
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
            class_ref = targeted_task["impl"]
            pipeline_class = get_class_from_dynamic_module(
                class_ref,
                model,
                code_revision=code_revision,
                **hub_kwargs,
            )
    else:
        normalized_task, targeted_task, _ = check_task(task)
        if pipeline_class is None:
            pipeline_class = targeted_task["impl"]

    # warning for not supported kwargs
    if "device_map" in kwargs:
        kwargs.pop("device_map", None)
        logger.warning("ms pipeline do not support keyword `device_map`, please check `device_id` & `device_target`.")
    if "device" in kwargs:
        kwargs.pop("device", None)
        logger.warning("ms pipeline do not support keyword `device`, please check `device_id` & `device_target`.")
    if "torch_dtype" in kwargs:
        kwargs.pop("torch_dtype", None)
        logger.warning("ms pipeline do not support keyword `torch_dtype`, please use `ms_dtype` instead.")

    # set mindspore context
    context_kwargs = {}
    if mode is None:
        logger.info("mindspore mode not set, using default graph mode.")
        mode = 0
    context_kwargs["mode"] = mode
    if device_id is not None:
        context_kwargs["device_id"] = device_id
    if device_target is not None:
        context_kwargs["device_target"] = device_target
    set_context(**context_kwargs)

    model_name = model if isinstance(model, str) else None

    # Load the correct model if possible
    if isinstance(model, str):
        model_classes = targeted_task.get("ms", (AutoModel,))
        model = load_model(
            model,
            model_classes=model_classes,
            config=config,
            task=task,
            **hub_kwargs,
            **model_kwargs,
        )

    model_config = model.config
    # pylint: disable=W0212
    hub_kwargs["_commit_hash"] = model.config._commit_hash
    # pylint: disable=C0123
    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None
    load_image_processor = type(model_config) in IMAGE_PROCESSOR_MAPPING or image_processor is not None

    if load_image_processor and load_feature_extractor:
        load_feature_extractor = False
    if (tokenizer is None
            and not load_tokenizer
            and normalized_task not in NO_TOKENIZER_TASKS
            # Using class name to avoid importing the real class.
            and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_tokenizer = True
    if (image_processor is None
            and not load_image_processor
            and normalized_task not in NO_IMAGE_PROCESSOR_TASKS
            # Using class name to avoid importing the real class.
            and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS
            and normalized_task != "automatic-speech-recognition"):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_image_processor = True
    if (feature_extractor is None
            and not load_feature_extractor
            and normalized_task not in NO_FEATURE_EXTRACTOR_TASKS
            # Using class name to avoid importing the real class.
            and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_feature_extractor = True

    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False

    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    if task in NO_IMAGE_PROCESSOR_TASKS:
        load_image_processor = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                if not isinstance(tokenizer[1], dict):
                    raise TypeError(f"tokenizer[1] should be a dict when type of tokenizer is tuple, "
                                    f"but get {type(tokenizer[1])}.")
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs.copy()

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )

    if load_image_processor:
        # Try to infer image processor from model or config name (if provided as str)
        if image_processor is None:
            if isinstance(model_name, str):
                image_processor = model_name
            elif isinstance(config, str):
                image_processor = config
            # Backward compatibility, as `feature_extractor` used to be the name
            # for `ImageProcessor`.
            elif feature_extractor is not None and isinstance(feature_extractor, BaseImageProcessor):
                image_processor = feature_extractor
            else:
                # Impossible to guess what is the right image_processor here
                raise Exception(
                    "Impossible to guess which image processor to use. "
                    "Please provide a PreTrainedImageProcessor class or a path/identifier "
                    "to a pretrained image processor."
                )

        # Instantiate image_processor if needed
        if isinstance(image_processor, (str, tuple)):
            image_processor = AutoImageProcessor.from_pretrained(
                image_processor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
            logger.warning("image_processor is a reserved field is currently not loaded.")

    if load_feature_extractor:
        # Try to infer feature extractor from model or config name (if provided as str)
        # Reserved branch
        logger.warning("feature_extractor is a reserved field is currently not loaded.")

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    if image_processor is not None:
        kwargs["image_processor"] = image_processor

    return pipeline_class(model=model, framework=framework, task=task, **kwargs)


def clean_custom_task(task_info):
    """clean custom task dict, transform default model from str to class name."""
    import mindformers

    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    ms_class_names = task_info.get("ms", ())
    if isinstance(ms_class_names, str):
        ms_class_names = [ms_class_names]
    task_info["ms"] = tuple(getattr(mindformers, c) for c in ms_class_names)
    return task_info, None


def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"text-to-audio"` (alias `"text-to-speech"` available)
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"`
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)


def load_model(model,
               config: any,
               model_classes: Optional[Dict[str, Tuple[type]]] = None,
               task: Optional[str] = None,
               **model_kwargs):
    """

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    Args:
        model (`str`, [`PreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        config ([`AutoConfig`]):
            The config associated with the model to help using the correct class
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        task (`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        Model : Model instance.
    """
    if isinstance(model, str):
        model_kwargs["_from_pipeline"] = task
        class_tuple = ()
        if model_classes:
            class_tuple = class_tuple + model_classes
        if config.architectures:
            classes = []
            for architecture in config.architectures:
                mindformers_module = importlib.import_module("mindformers")
                module_class = getattr(mindformers_module, architecture, None)
                if module_class is not None:
                    classes.append(module_class)
            class_tuple = class_tuple + tuple(classes)

        if not class_tuple:
            raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")

        all_traceback = {}
        for model_class in class_tuple:
            kwargs = model_kwargs.copy()
            try:
                model = model_class.from_pretrained(model, **kwargs)
                if hasattr(model, "eval"):
                    model = model.eval()
                # Stop loading on the first successful load.
                break
            except (OSError, ValueError):
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue

        if isinstance(model, str):
            error = ""
            for class_name, trace in all_traceback.items():
                error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
            raise ValueError(
                f"Could not load model {model} with any of the following classes: {class_tuple}."
                f" See the original errors:\n\n{error}\n"
            )
    return model


def is_experimental_mode(model, **kwargs):
    """Check whether pipeline should go into original or experimental mode."""
    experimental_keys = ["config", "feature_extractor", "framework", "revision", "use_fast", "token",
                         "trust_remote_code", "model_kwargs", "pipeline_class"]
    # model instance, go into original mode
    if not isinstance(model, str):
        return False
    # model str with repo name, go into experimental mode
    if not os.path.exists(model) and "/" in model and model.split("/")[0] != "mindspore":
        return True
    # model str with model directory, go into experimental mode
    if os.path.isdir(model):
        return True
    # in other cases, should go into original mode
    # if still got keys in experimental api, raise error
    experimental_keys.extend(kwargs.keys())
    tmp_dict = {}.fromkeys(experimental_keys)
    if len(tmp_dict) < len(experimental_keys):
        raise ValueError(f"In pipeline api, got model with str: \"{model}\", should use origin mode. "
                         f"But got kwargs {kwargs}, in which some items is only support in experimental_mode. "
                         "Please recheck your input args of pipeline api.")
    # with model str, go into original mode
    return False
