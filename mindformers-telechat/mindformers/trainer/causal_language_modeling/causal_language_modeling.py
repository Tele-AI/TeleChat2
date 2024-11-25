# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Causal Image Modeling Trainer."""
import os
import time
import datetime
from typing import Optional, List, Union
from pprint import pprint

import numpy as np
from mindspore import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase, build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from mindformers.tools.check_rules import check_rules
from mindformers.models.auto import AutoModel
from mindformers.mindformer_book import MindFormerBook

from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer
from ..utils import transform_and_load_checkpoint, get_real_rank

GENERATE_METRIC_NAMES = ['ADGENMetric', 'EmF1Metric']
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class CausalLanguageModelingTrainer(BaseTrainer):
    """
    CausalLanguageModelingTrainer Task For Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None

    Examples:
        >>> from mindformers import CausalLanguageModelingTrainer
        >>> gen_trainer = CausalLanguageModelingTrainer(model_name="gpt2")
        >>> type(gen_trainer)
        <class 'mindformers.trainer.causal_language_modeling.causal_language_modeling.CausalLanguageModelingTrainer'>
    """

    def __init__(self, model_name: str = None):
        super(CausalLanguageModelingTrainer, self).__init__("text_generation", model_name)

    @staticmethod
    def check_generate_evaluate(metric_name_list):
        """whether use generate evaluate"""
        use_generate_evaluate = False
        if len(metric_name_list) == 1:
            metric_name = metric_name_list[0]
            if metric_name in GENERATE_METRIC_NAMES:
                use_generate_evaluate = True
        else:
            for metric_name in metric_name_list:
                if metric_name in GENERATE_METRIC_NAMES:
                    use_generate_evaluate = True
                elif use_generate_evaluate:
                    raise ValueError(f"The metric type can't include both {GENERATE_METRIC_NAMES} \
                        and other metric type at the same time.")
        return use_generate_evaluate

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Train task for CausalLanguageModeling Trainer.
        This function is used to train or fine-tune the network.
        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer.It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]):
                The training dataset.It support real dataset path or BaseDateset class or MindSpore
                Dataset class. Default: None.
            wrapper (Optional[TrainOneStepCell]):
                Wraps the `network` with the `optimizer`.It support TrainOneStepCell class of MindSpore.
                Default: None.
            optimizer (Optional[Optimizer]):
                he training network's optimizer. It support Optimizer class of MindSpore. Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The training callback function. It support CallBack or CallBack List of MindSpore. Default: None.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                 network: Optional[Union[Cell, PreTrainedModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """
        Evaluate task for CausalLanguageModeling Trainer. This function is used to evaluate the network.
        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, dataset, callbacks, compute_metrics.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset]]):
                The evaluate dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The eval callback function. It supports CallBack or CallBack List of MindSpore. Default: None.
            compute_metrics (Optional[Union[dict, set]]):
                The metric of evaluating. It supports dict or set in MindSpore's Metric class. Default: None.
        """
        metric_name_list = [metric['type'] for metric in config.metric]
        if len(metric_name_list) == 1:
            kwargs.setdefault("metric_name", metric_name_list[0])
        use_generate_evaluate = self.check_generate_evaluate(metric_name_list)

        if use_generate_evaluate:
            self.generate_evaluate(
                config,
                network=network,
                dataset=dataset,
                compute_metrics=compute_metrics,
                **kwargs)
        else:
            self.evaluate_process(
                config=config,
                network=network,
                dataset=dataset,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                **kwargs)

    def generate_evaluate(self,
                          config,
                          network=None,
                          dataset=None,
                          compute_metrics=None,
                          tokenizer=None,
                          **kwargs):
        r"""Evaluate the text generate task. Return metrics with Rouge-1, Rouge-2, Rouge-l and BLEU. """
        self.eval_dataset = dataset if dataset else self.eval_dataset
        metric_name = kwargs.get("metric_name")
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        enable_max_new_tokens = bool(config.model.model_config.max_new_tokens)
        # it does not support max_new_tokens as input parameter in text_generate, so reset batch_size to 1 when the
        # follow scenario happens
        if metric_name == "EmF1Metric" and enable_max_new_tokens and config.runner_config.batch_size != 1:
            logger.info("For metric %s, it only supports batch size equals 1, so reset batch size to 1 here.",
                        metric_name)
            config.runner_config.batch_size = 1

        # build dataset
        logger.info(".........Build Dataset For Evaluate..........")
        dataset = self.create_eval_dataset()
        self.set_eval_dataset(dataset)
        logger.info("Create evaluate dataset finish, dataset size:%d", dataset.get_dataset_size())

        # check rules
        check_rules(config, mode='eval', network=network, dataset=dataset, task=self.task)

        # build network
        if network is None:
            network = self.create_network(default_args={"parallel_config": config.parallel_config,
                                                        "moe_config": config.moe_config})
        self.set_network(network, is_train=False)

        self.count_parameters()

        # build metric
        logger.info(".........Build Compute Metrics For Evaluate..........")
        if compute_metrics is None:
            compute_metrics = self.create_metrics(metric_name=metric_name)
            for metric_name in compute_metrics:
                compute_metrics[metric_name].clear()

        # build tokenizer
        logger.info(".........Build tokenizer For Evaluate..........")
        if tokenizer is None and config.processor.tokenizer:
            tokenizer = build_tokenizer(config.processor.tokenizer)

        logger.info(".........Starting Init Evaluate Model..........")
        model = Model(network, eval_network=network)

        if config.load_checkpoint or config.only_save_strategy:
            if config.load_checkpoint in SUPPORT_MODEL_NAMES:
                config.load_checkpoint = \
                    AutoModel.from_pretrained(config.load_checkpoint).default_checkpoint_download_path
            logger.info(".............Start load checkpoint for eval..................")

            dataset_dict = next(dataset.create_dict_iterator())
            input_ids = dataset_dict['input_ids'].asnumpy()
            labels = dataset_dict['labels'].asnumpy()
            infer_data = network.prepare_inputs_for_predict_layout(input_ids, labels=labels)
            transform_and_load_checkpoint(config, model, network, infer_data, do_eval=True)

        logger.info('.........Starting Evaluate Model..........')
        if get_real_rank() % 8 == 0:
            pprint(config)
        # generate config
        do_sample = config.model.model_config.do_sample
        top_p = config.model.model_config.top_p
        top_k = config.model.model_config.top_k
        max_length = config.model.model_config.max_decode_length

        total_tokens_num = 0
        total_time = 0.0001
        pad_token_id = tokenizer.pad_token_id
        len_dataset = dataset.get_dataset_size()
        for i, inputs in enumerate(dataset.create_dict_iterator()):
            input_ids = inputs['input_ids'].asnumpy()
            labels = inputs['labels'].asnumpy()

            valid_length_each_example = []
            for j in range(input_ids.shape[0]):
                # As the nonzero returns the index and we need length
                valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
            valid_length_each_example = np.array(valid_length_each_example)

            if enable_max_new_tokens:
                # When we act as it, the batch_size is 1. it will be replaced when text_generator supports batch_size
                # inference quickly or text_generator supports max_new_tokens as the input parameter.
                max_length = valid_length_each_example[0] + self.config.model.model_config.max_new_tokens

            start_time = time.time()
            outputs = model.predict_network.generate(input_ids, do_sample=do_sample, max_length=max_length,
                                                     top_p=top_p, top_k=top_k, return_dict_in_generate=False)
            output_ids = []
            for j in range(input_ids.shape[0]):
                output_ids.append(outputs[j][int(valid_length_each_example[j]):])
            end_time = time.time()
            avg_cost_time = (end_time - start_time) / input_ids.shape[0]

            tokens_num = 0
            for batch_index, _ in enumerate(output_ids):
                tokens_num += output_ids[batch_index].shape[0]
            if i != 0:
                total_tokens_num += tokens_num
                total_time += end_time - start_time

            # compute time remaining
            avg_time = total_time / (i + 1)
            remain_time = (len_dataset - i - 1) * avg_time
            logger.info(f"Step[{i + 1} / {len_dataset}], cost time {end_time - start_time:.4f}s, " +
                        f"every example cost time is {avg_cost_time:.4f}, " +
                        f"generate speed: {tokens_num / (end_time - start_time):.4f} tokens/s, " +
                        f"avg speed: {total_tokens_num / total_time:.4f} tokens/s, "
                        f"remaining time: {datetime.timedelta(seconds=int(remain_time))}")

            # decode input_id and label to string
            pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
            labels_str = tokenizer.decode(labels, skip_special_tokens=True)

            for k in compute_metrics:
                compute_metrics[k].update(pres_str, labels_str)

        for k in compute_metrics:
            compute_metrics[k].eval()

        logger.info('...........Evaluate Over!...............')

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                **kwargs):
        """
        Executes the predict of the trainer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]):
                The predict data. It supports 1) a text string to be translated, 1) a file name where each
                line is a text to be translated  and 3) a generator dataset. Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]):
                The tokenizer for tokenizing the input text. Default: None.

        Returns:
            List, a list of prediction.
        """
        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list, GeneratorDataset)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list, GeneratorDataset], but got type {type(input_data)}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, 'r') as fp:
                input_data_list = []
                for line in fp:
                    line = line.strip('\n')
                    line = line.replace(r"\n", "\n")
                    input_data_list.append(line)
            input_data = input_data_list

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='text_generation',
                                    network=network,
                                    tokenizer=tokenizer,
                                    **kwargs)

    def _evaluate_in_training(self, model, eval_dataset):
        logger.info('Starting Evaluate Model')
        metric_name_list = list(self.compute_metrics.keys())
        use_generate_evaluate = self.check_generate_evaluate(metric_name_list)

        if use_generate_evaluate:
            config = self.config
            dataset = eval_dataset
            model = model.eval_network
            enable_max_new_tokens = bool(config.model.model_config.max_new_tokens)

            # build metric
            compute_metrics = self.compute_metrics
            for k in compute_metrics:
                compute_metrics[k].clear()

            # build tokenizer
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                logger.info("Build tokenizer For Evaluate")
                self.tokenizer = build_tokenizer(config.processor.tokenizer)

            self.set_network(model, is_train=False)

            # generate config
            do_sample = config.model.model_config.do_sample
            top_p = config.model.model_config.top_p
            top_k = config.model.model_config.top_k
            max_length = config.model.model_config.max_decode_length

            total_tokens_num = 0
            total_time = 0.0001
            pad_token_id = self.tokenizer.pad_token_id
            len_dataset = dataset.get_dataset_size()
            for i, inputs in enumerate(dataset.create_dict_iterator()):
                input_ids = inputs['input_ids'].asnumpy()
                labels = inputs['labels'].asnumpy()

                valid_length_each_example = []
                for j in range(input_ids.shape[0]):
                    # As the nonzero returns the index and we need length
                    valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
                valid_length_each_example = np.array(valid_length_each_example)

                if enable_max_new_tokens:
                    # When we act as it, the batch_size is 1. it will be replaced when text_generator supports batch_size
                    # inference quickly or text_generator supports max_new_tokens as the input parameter.
                    max_length = valid_length_each_example[0] + self.config.model.model_config.max_new_tokens

                start_time = time.time()
                outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length,
                                         top_p=top_p, top_k=top_k, return_dict_in_generate=False)
                output_ids = []
                for j in range(input_ids.shape[0]):
                    output_ids.append(outputs[j][int(valid_length_each_example[j]):])
                end_time = time.time()
                avg_cost_time = (end_time - start_time) / input_ids.shape[0]

                tokens_num = 0
                for batch_index, _ in enumerate(output_ids):
                    tokens_num += output_ids[batch_index].shape[0]
                if i != 0:
                    total_tokens_num += tokens_num
                    total_time += end_time - start_time

                # compute time remaining
                avg_time = total_time / (i + 1)
                remain_time = (len_dataset - i - 1) * avg_time
                logger.info(f"Step[{i + 1} / {len_dataset}], cost time {end_time - start_time:.4f}s, " +
                            f"every example cost time is {avg_cost_time:.4f}, " +
                            f"generate speed: {tokens_num / (end_time - start_time):.4f} tokens/s, " +
                            f"avg speed: {total_tokens_num / total_time:.4f} tokens/s, "
                            f"remaining time: {datetime.timedelta(seconds=int(remain_time))}")

                # decode input_id and label to string
                pres_str = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                labels_str = self.tokenizer.decode(labels, skip_special_tokens=True)

                for k in compute_metrics:
                    compute_metrics[k].update(pres_str, labels_str)

            score_dict = {}
            for k in compute_metrics:
                score_dict[k] = compute_metrics[k].eval()

            self.set_network(model, is_train=True)

            logger.info('Evaluate Over!.....')
            output = score_dict
        else:
            output = super()._evaluate_in_training(model=model, eval_dataset=eval_dataset)
        return output
