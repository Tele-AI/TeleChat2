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
"""public modules for testing models."""
import os
from functools import partial
import numpy as np

from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer, TrainingArguments
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.models.auto import AutoTokenizer

from tests.st.training_checker import TrainingChecker

def create_tokenizer(auto_tokenizer=None):
    """build tokenizer from name."""
    if auto_tokenizer is None:
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        vocab_file = f"{cur_dir}/llama2_tokenizer/tokenizer.model"
        assert os.path.exists(vocab_file), f"vocab_file {vocab_file} not exists."

        tokenizer_config = {
            'unk_token': '<unk>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'pad_token': '<unk>',
            'type': 'LlamaTokenizer',
            'vocab_file': vocab_file}
        tokenizer = build_tokenizer(config=tokenizer_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(auto_tokenizer)
    return tokenizer


class ModelTester:
    """
    A public class for testing models.

    Args:
        run_mode (str):
            The current run mode, support 'train', 'predict', 'eval', 'finetune'.
            Notice that there is no difference between 'train' and 'finetune' currently.
        batch_size (int):
            Specify the batch size for training and inference.
            For inference, it will affect the batch size of the model.
        step_num (int, optional):
            Number of steps for model training. Defaults to 20.
        num_train_epochs (int, optional):
            Number of epochs for model training. Defaults to 1.
        use_label (bool, optional):
            If return generated label in dataset, will affect data generation in 'train' or 'eval' mode.
            Defaults to False.
        experiment_mode (bool, optional):
            This parameter affects the training and inference functions. Defaults to False.
            For training, in experimental mode 'loss_std' and 'avg_time_std' will be printed.
            For inference, in experimental mode 'expect_outputs' will be printed.
        tokenizer (str, optional):
            The type of tokenizer used in testing. Defaults to 'gpt2'.

    Raises:
        AssertionError
    """

    def __init__(self,
                 run_mode: str,
                 batch_size: int,
                 step_num: int = 20,
                 num_train_epochs: int = 1,
                 use_label: bool = False,
                 experiment_mode: bool = False,
                 **kwargs):
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        set_seed(0)
        np.random.seed(0)

        self.batch_size = batch_size
        self.step_num = step_num
        self.use_label = use_label
        self.experiment_mode = experiment_mode

        optimizer_config = {
            # optimizer config
            'optim': 'fp32_adamw',
            'adam_beta1': 0.9,
            'adam_beta2': 0.95,
            'adam_epsilon': 1.e-8,
            # lr schedule config
            'lr_scheduler_type': 'cosine',
            'learning_rate': 2.e-5,
            'lr_end': 1.e-6,
            'warmup_steps': 0
        }
        kwargs.update(optimizer_config)

        self.args = TrainingArguments(
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            **kwargs)

        support_run_mode = ['train', 'predict', 'eval', 'finetune']
        assert run_mode in support_run_mode, f"run_mode should in {support_run_mode}, but got {run_mode}."

        self.run_mode = run_mode
        if self.run_mode == 'eval':
            self.step_num = 1

    @staticmethod
    def generate_data(seq_len, vocab_size, batch_size=4, step_num=20,
                      use_label=False, is_eval=False):
        """generate data for testing model."""
        input_ids = np.random.randint(
            low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)

        if is_eval:  # set pad_token_id in generated data
            for i in range(step_num * batch_size):
                input_ids[i][seq_len // 2:] = 0

        for input_data in input_ids:
            if use_label:
                yield input_data, input_data
            else:
                yield input_data

    def get_dataset(self, config):
        """build dataset for model training."""
        if self.run_mode == 'train':
            seq_length = config.seq_length + 1
        else:
            seq_length = config.seq_length

        prepare_data = partial(self.generate_data,
                               seq_len=seq_length,
                               vocab_size=config.vocab_size,
                               batch_size=self.batch_size,
                               step_num=self.step_num,
                               use_label=self.use_label,
                               is_eval=self.run_mode == 'eval')

        if self.use_label:
            dataset = GeneratorDataset(prepare_data, column_names=['input_ids', 'labels'])
        else:
            dataset = GeneratorDataset(prepare_data, column_names=['input_ids'])
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def set_train(self, model, config, dataset=None, loss_std=None, avg_time_std=None,
                  checker_config=None, parallel_config=None, task='text_generation'):
        """set model train."""
        if not self.experiment_mode:
            assert isinstance(loss_std, list) and self.step_num == len(loss_std)
            assert isinstance(avg_time_std, (int, float)) or avg_time_std is None

        dataset = self.get_dataset(config) if dataset is None else dataset

        if checker_config is None:
            checker_config = dict()
        callback = TrainingChecker(loss_list_std=loss_std,
                                   avg_step_time_std=avg_time_std,
                                   experiment_mode=self.experiment_mode,
                                   **checker_config)

        task_trainer = Trainer(task=task,
                               model=model,
                               args=self.args,
                               train_dataset=dataset,
                               callbacks=callback)

        task_trainer.config.runner_config.epochs = 1
        task_trainer.config.runner_config.sink_mode = False
        task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
        task_trainer.config.callbacks = task_trainer.config.callbacks[:1]

        if parallel_config is not None:
            task_trainer.set_parallel_config(**parallel_config)

        task_trainer.train()
        if self.experiment_mode:
            callback.get_experiment_results()

    def set_predict(self, model, predict_data='hello world.', expect_outputs='', auto_tokenizer=None):
        """set model predict."""
        tokenizer = create_tokenizer(auto_tokenizer)
        task_trainer = Trainer(task='text_generation',
                               model=model,
                               args=self.args,
                               tokenizer=tokenizer)
        predict_data = [predict_data] * self.batch_size
        outputs = task_trainer.predict(input_data=predict_data,
                                       max_length=20,
                                       repetition_penalty=1,
                                       top_k=3,
                                       top_p=1,
                                       batch_size=self.batch_size)
        outputs = outputs[0]['text_generation_text']
        print(outputs)

        if not self.experiment_mode:
            expect_outputs = [expect_outputs] * self.batch_size
            assert outputs == expect_outputs

        return outputs[0]

    def set_eval(self, model, config, metric='ADGENMetric', tokenizer_config=None):
        """set model evaluate.

        evaluate function of models have 2 branches:
        1. forward generate (adapted kbk infer), only support 'ADGENMetric', 'EmF1Metric' metric
        2. forward mindspore.model.eval (error in kbk infer)
        """
        metric = {'type': metric}
        dataset = self.get_dataset(config)

        tokenizer = create_tokenizer(tokenizer_config)
        task_trainer = Trainer(task='text_generation',
                               model=model,
                               args=self.args,
                               eval_dataset=dataset,
                               tokenizer=tokenizer)
        task_trainer.config.metric = metric

        task_trainer.evaluate()
