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
"""The base class that initialise the model and some public functions needed by inference tool and pretrain tool"""
import os
import sys
from glob import glob
from pathlib import Path

from mindspore import Model

import mindformers
from mindformers.model_runner import register_auto_class
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.auto.modeling_auto import AutoModel
from mindformers.pet import get_pet_model
from mindformers.tools.logger import logger
from mindformers import MindFormerConfig

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(mindformers.__file__))


def get_parallel_status():
    device_num = os.getenv('MS_WORKER_NUM')
    if device_num:
        return True, int(device_num)
    return False, 1


def convert_path(src_path, save_path):
    src_path = os.path.realpath(src_path)
    file_name = os.path.basename(src_path)
    file_path = os.path.join(save_path, file_name)
    return src_path, file_path


class BaseInitModel:
    """
    The base class that initialise the model

    Args:
        model_name_or_dir: str = 'llama2_7b',
        predict_data=None,
        input_args
    """

    def __init__(self, model_name_or_dir: str = 'llama2_7b', input_args=None):

        self.model_name_or_dir = model_name_or_dir
        self.input_args = input_args

        if Path(model_name_or_dir).is_dir():
            # research offline
            self.model_path = self.model_name_or_dir
            yaml_files = glob(str(Path(model_name_or_dir) / '*.yaml'))
            if yaml_files:
                self.config = MindFormerConfig(yaml_files[0])
                self.model_name = self.config.trainer.model_name
            else:
                raise RuntimeError('Cannot find yaml file')
            sys.path.append(str(Path(self.model_name_or_dir).absolute()))
            register_auto_class(self.config, self.model_name_or_dir, 'AutoConfig')
            register_auto_class(self.config, self.model_name_or_dir, 'AutoTokenizer')
            register_auto_class(self.config, self.model_name_or_dir, 'AutoModel')
        else:
            # online
            self.model_name = self.model_name_or_dir
            self.model_prefix = self.model_name_or_dir.split('_')[0]
            self.config = AutoConfig.from_pretrained(self.model_name_or_dir)

        # prepare config
        self.custom_model = False
        self.use_pet_model = False
        self.init_config()

        # build model
        logger.info("build base model.")

        self.network = AutoModel.from_config(self.config)
        # self.network = build_model(self.config.model)
        if self.use_pet_model:
            self.build_pet_model()
        self.model = Model(self.network)

    def build_pet_model(self):
        """Build pet model"""
        logger.info("build pet model.")
        pet_config = self.config.model.model_config.pet_config
        self.network = get_pet_model(self.network, pet_config)

    def init_config(self):
        """Init config"""
        # process parallel config
        self.process_parallel_config()

        # process model config
        self.process_model_config()

        # set auto transform ckpt
        auto_trans_ckpt = self.config.use_parallel and os.path.isfile(self.config.load_checkpoint)
        self.config.auto_trans_ckpt = self.input_args.get('auto_trans_ckpt', auto_trans_ckpt)
        self.config.model.model_config.batch_size = self.input_args.get(
            'data_batch_size',
            self.config.model.model_config.batch_size
        )
        self.config.model.model_config.seq_length = self.input_args.get(
            'data_seq_len',
            self.config.model.model_config.seq_length
        )
        self.config.model.model_config.num_layers = self.input_args.get(
            'model_num_layers',
            self.config.model.model_config.num_layers
        )
        self.config.src_strategy_path_or_dir = self.input_args.get(
            'src_strategy',
            self.config.src_strategy_path_or_dir
        )

    def process_parallel_config(self):
        """Process parallel config"""
        dp = self.input_args.get('model_parallel', None)
        mp = self.input_args.get('data_parallel', None)
        if dp and mp:
            self.config.parallel_config.model_parallel = dp
            self.config.parallel_config.data_parallel = mp


    def process_model_config(self):
        """Process model config"""
        self.config.model.model_config.checkpoint_name_or_path = None
        self.config.model.model_config.parallel_config = self.config.parallel_config

        if getattr(self.config.model.model_config, 'pet_config'):
            self.use_pet_model = True

        for k, v in self.input_args.items():
            if self.config.model.model_config.get(k):
                self.custom_model = True
                self.config.model.model_config[k] = v

    def process_checkpoint(self):
        """Process checkpoint"""
        ckpt_files = glob(f"{self.model_path}/*.ckpt")
        load_checkpoint = self.input_args.get('load_checkpoint', None)

        if load_checkpoint == "" or load_checkpoint is False:
            # not load checkpoint later
            logger.warning(f"set load checkpoint: False. Running [pretrain].")
            self.config.load_checkpoint = ""
            return

        if not load_checkpoint or not os.path.exists(load_checkpoint):
            load_checkpoint = self.config.load_checkpoint

        if load_checkpoint and os.path.exists(load_checkpoint):
            # make soft link of checkpoint file in model_path
            src_path, load_checkpoint = convert_path(load_checkpoint, self.model_path)
            try:
                os.symlink(src_path, load_checkpoint)
            except OSError:
                logger.warning(f"link of {src_path} -> {load_checkpoint} is existed.")
            logger.info(f"Checkpoint found. Running [finetune] with {load_checkpoint}.")
        elif ckpt_files:  # use first checkpoint file in model_path
            if not self.use_pet_model or len(ckpt_files) != 2:
                logger.info(f"Using PET model and found 2 ckpt files in {self.model_path}.")
                load_checkpoint = self.model_path
            else:
                logger.info(f"Found {len(ckpt_files)} ckpt files in {self.model_path}.")
                load_checkpoint = ckpt_files[0]
            logger.info(f"Running [finetune] with checkpoint {load_checkpoint}.")
        else:
            logger.warning(f"Currently does not support downloading the ckpt file of {self.model_name} from online.")
            logger.info(f"Running [pretrain] as no checkpoint was loaded.")

        logger.info(f"Using checkpoint file: {load_checkpoint}.")

        self.config.load_checkpoint = load_checkpoint
