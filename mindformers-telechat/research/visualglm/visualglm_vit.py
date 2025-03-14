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
# This file was refer to project:
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""vit models for visualglm"""
import os
from collections import OrderedDict

from mindspore import load_checkpoint
import mindspore.common.dtype as mstype

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.vit.vit import ViTModel, ViTConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.tools.utils import try_sync_file


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ViTModelForBlip2(ViTModel):
    """
    ViTModel For visualglm Models, loading a pretrained weight.
    forward will return the penultimate output.
    """
    _support_list = MindFormerBook.get_config_support_list()['vit']

    def __init__(self, config: ViTConfig):
        super(ViTModelForBlip2, self).__init__(config)
        print(f"------------------vit checkpoint path: {config.checkpoint_name_or_path}")
        self.load_checkpoint(config)

    def construct(self, image):
        return self.construct_without_pool(image)

    def load_checkpoint(self, config: ViTConfig):
        """
        load checkpoint for BertLMHeadModel. (we can use the param for BertModel on obs,
        but we need to alter the names of some param)

        Args:
            config (ModelConfig): QFormerConfig instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        # the relevant file will be downloaded from the Obs platform.
        if not os.path.exists(checkpoint_name_or_path):
            if checkpoint_name_or_path not in self._support_list:
                raise ValueError(f"{checkpoint_name_or_path} is not a supported default model"
                                 f" or a valid path to checkpoint,"
                                 f" please select from {self._support_list}.")
            # on Atlas 800T A2, load the 'resized' checkpoint.
            if not config.resize_token_embeddings and not checkpoint_name_or_path.endswith("_resized"):
                checkpoint_name_or_path = checkpoint_name_or_path + "_resized"
            checkpoint_name = checkpoint_name_or_path
            default_checkpoint_download_folder = os.path.join(
                MindFormerBook.get_default_checkpoint_download_folder(),
                checkpoint_name_or_path.split("_")[0])
            if not os.path.exists(default_checkpoint_download_folder):
                os.makedirs(default_checkpoint_download_folder, exist_ok=True)

            ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name + ".ckpt")
            if not os.path.exists(ckpt_file):
                url = MindFormerBook.get_moddownload_with_progress_barel_ckpt_url_list()[checkpoint_name_or_path][0]
                succeed = (url, ckpt_file)
                if not succeed:
                    logger.info("checkpoint download failed, and pretrained weights are unloaded.")
                    return
            try_sync_file(ckpt_file)
            self.default_checkpoint_download_path = ckpt_file
            logger.info("start to read the ckpt file: %s", os.path.getsize(ckpt_file))
        else:
            ckpt_file = checkpoint_name_or_path
        param = load_checkpoint(ckpt_file)
        try:
            self.convert_vit_model_params(param)
            logger.info("weights in %s are loaded", ckpt_file)
        except RuntimeError:
            logger.error("the given config and weights in %s are"
                         " mismatched, and weights load failed", ckpt_file)

    def convert_vit_model_params(self, vit_model_params: OrderedDict):
        """
        convert params from BertModel in MindFormers, some param names are altered.
        """
        param_dict = self.parameters_dict()
        for name, data in param_dict.items():
            if name.startswith('ln_vision'):
                new_name = name
            else:
                new_name = 'visual_encoder.' + name
            if new_name not in vit_model_params:
                logger.warning("%s does not exist", new_name)
                continue
            new_data = vit_model_params[new_name]
            new_data = new_data.astype(mstype.float32)
            data.assign_value(new_data)
