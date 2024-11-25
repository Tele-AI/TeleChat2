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
"""Image-to-text Retrieval Trainer."""
from typing import List, Optional, Union
from pprint import pprint

import numpy as np
from mindspore import dtype as mstype
from mindspore.train import Callback

from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_network, PreTrainedModel
from mindformers.core.callback import build_callback
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params, get_real_rank
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.check_rules import check_rules
from .eval_utils import compute_itm_scores, extract_image_text_mapping, \
    prepare_inputs_for_itm_eval, report_metrics
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="image_to_text_retrieval")
class ImageToTextRetrievalTrainer(BaseTrainer):
    """
    Image-to-text Retrieval Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    """
    def __init__(self, model_name: str = None):
        super(ImageToTextRetrievalTrainer, self).__init__("image_to_text_retrieval", model_name)
        self.model_name = model_name
        self.kwargs = None

    def train(self, **kwargs):
        raise NotImplementedError(
            "The image to text retrieval task does not support train.")

    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, PreTrainedModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 **kwargs):
        """
        Evaluation task for ImageToTextRetrievalTrainer Trainer.

        Args:
            config (Optional[Union[dict, ConfigArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or ConfigArguments class. Default: None.
            network (Optional[Union[str, PreTrainedModel]]):
                The network for trainer. It supports model name or MindSpore Cell class. Default: None.
            dataset (Optional[Union[str, GeneratorDataset]]):
                The training dataset. It support real dataset path or MindSpore Dataset class. Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The training callback function. It support CallBack or CallBack List of MindSpore. Default: None.
        """
        self.kwargs = kwargs
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)

        # check rules
        check_rules(config, mode='eval', network=network, dataset=dataset)

        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task)
        logger.info("Create eval dataset finish, dataset size:%d", dataset.get_dataset_size())

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_network(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config,
                "is_training": False})

        network = network.to_float(mstype.float16)

        # checkpoint
        logger.info(".........Loading Checkpoint..........")
        if config.model.model_config.checkpoint_name_or_path is not None:
            network.load_checkpoint(config.model.model_config)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.eval_callbacks))

        logger.info(".........Starting Evaling Model..........")
        if get_real_rank() % 8 == 0:
            pprint(config)

        # k_test value, for topk
        k_test = config.eval_dataset.k_test if \
            config.eval_dataset.k_test is not None else 128
        # trainer arguments overrides top_k
        k_test = kwargs.pop('k_test', k_test)
        logger.info("========= k_text num: %d =========", k_test)

        # whether adding additional itm score
        add_extra_itm_score = config.eval_dataset.add_extra_itm_score
        # trainer arguments overrides add_extra_itm_score
        add_extra_itm_score = kwargs.pop('add_extra_itm_score', add_extra_itm_score)

        # prepare inputs for computing simliarity matrix
        image_feats, text_feats, vit_outputs, text_ids = prepare_inputs_for_itm_eval(network, dataset)
        logger.info("prepare_inputs_for_itm_eval finished.")

        # compute image-to-text/text-to-image similarity scores
        sims_matrix, vit_outputs, text_ids = network(image_feats,
                                                     text_feats,
                                                     vit_outputs,
                                                     text_ids,
                                                     add_extra_itm_score=add_extra_itm_score)
        logger.info("sims_matrix computed.")

        score_i2t, score_t2i = compute_itm_scores(network,
                                                  sims_matrix.asnumpy(),
                                                  vit_outputs.asnumpy(),
                                                  text_ids.asnumpy(),
                                                  k_test,
                                                  add_extra_itm_score)

        # ground-truth image-text mapping
        img2txt, txt2img = extract_image_text_mapping(dataset, score_i2t, score_t2i)

        # ground-truth type validation
        if not isinstance(img2txt, (np.ndarray, list, dict)) or not isinstance(txt2img, (np.ndarray, list, dict)):
            raise ValueError("img2txt and txt2img should both be numpy.ndarray, list or dict.")

        # report evaluation results
        eval_result = report_metrics(
            score_i2t,
            score_t2i,
            img2txt,
            txt2img,
        )

        logger.info(eval_result)
        logger.info(".........Evaluate Over!.............")
        return eval_result
