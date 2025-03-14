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
"""visualglm finetune runner. """
import argparse
import json

from mindformers import MindFormerConfig, MindFormerRegister, MindFormerModuleType, build_context
from mindformers.tools.utils import str2bool
from mindformers import Trainer
from mindformers.tools.logger import logger

from visualglm import VisualglmWithLora
from visualglm_config import VisualGLMConfig
from visualglm_dataloader import VisualGLMDataLoader
from visualglm_dataset import VisualGLMDataset
from visualglm_lr_schedule import AnnealingLR


def main(args):
    mode = args.graph_mode

    config_path = args.config_path
    mindformer_config = MindFormerConfig(config_path)

    if mode is not None:
        mindformer_config.context.mode = mode

    if args.device_id != -1:
        mindformer_config.context.device_id = args.device_id

    if args.device_target:
        mindformer_config.context.device_target = args.device_target

    # init_context(mindformer_config, device_id=args.device_id, device_target=args.device_target, mode=mode)

    build_context(mindformer_config)

    logger.info(f"--------------- mindformer_config: {mindformer_config}")

    model_config = VisualGLMConfig.from_pretrained(args.config_path)
    model_config.max_txt_len = args.seq_length

    if args.checkpoint is not None:
        logger.info(f"checkpoint: {args.checkpoint}")
        model_config.checkpoint_name_or_path = args.checkpoint

    init_batch_size(args, mindformer_config, model_config)

    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    MindFormerRegister.register_cls(
        AnnealingLR, module_type=MindFormerModuleType.LR, alias="AnnealingLR")

    MindFormerRegister.register_cls(
        VisualGLMDataLoader, module_type=MindFormerModuleType.DATASET_LOADER, alias="VisualGLMDataLoader")

    MindFormerRegister.register_cls(
        VisualGLMDataset, module_type=MindFormerModuleType.DATASET, alias="VisualGLMDataset")

    dataset_dir = mindformer_config.train_dataset.data_loader.dataset_dir
    logger.info(f"------------------------- dataset_dir: {dataset_dir}")
    with open(dataset_dir) as dataset_file:
        datasets = json.load(dataset_file)
    data_size = len(datasets)
    logger.info(f"------------------------ data_size: {data_size}")

    num_iters = mindformer_config.lr_schedule.num_iters
    batch_size = model_config.batch_size
    data_parallel = 1
    if mindformer_config.use_parallel:
        data_parallel = mindformer_config.parallel_config.data_parallel

    scale = num_iters * batch_size * data_parallel // data_size + 1

    logger.info(f"dataset scale: {scale} = {num_iters} * {batch_size} * {data_parallel} // {data_size} + 1")
    mindformer_config.train_dataset.data_loader.scale = scale
    mindformer_config.train_dataset_task.dataset_config.data_loader.scale = scale

    train_dataset = VisualGLMDataset(mindformer_config.train_dataset_task.dataset_config)

    model = VisualglmWithLora(model_config)
    task = Trainer(args=mindformer_config,
                   model=model,
                   model_name='visualglm_6b',
                   task='text_generation',
                   train_dataset=train_dataset,
                   pet_method='')

    task.train(train_checkpoint=mindformer_config.load_checkpoint,
               auto_trans_ckpt=mindformer_config.auto_trans_ckpt, resume_training=False)


def init_batch_size(args, mindformer_config, model_config):
    if args.batch_size > 1:
        model_config.batch_size = args.batch_size
    else:
        model_config.batch_size = 1
    mindformer_config.runner_config.batch_size = model_config.batch_size
    mindformer_config.model.model_config.batch_size = model_config.batch_size
    model_config.text_config.batch_size = model_config.batch_size
    mindformer_config.train_dataset.batch_size = model_config.batch_size
    mindformer_config.train_dataset_task.dataset_config.batch_size = model_config.batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_mode', default=0, type=int, required=False, help='graph mode')
    parser.add_argument('--model_type', default="visualglm_6b", type=str, required=False, help='model type')
    parser.add_argument('--config_path', default="run_visualglm_lora.yaml", type=str, required=False,
                        help='config path')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='device id')
    parser.add_argument('--device_target', type=str, default='Ascend', required=False, help='device target')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch_size')
    parser.add_argument('--checkpoint', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--use_past', type=str2bool, default=None, required=False, help='whether use past')
    parser.add_argument('--do_sample', type=str2bool, default=False, required=False, help='whether do sample')
    parser.add_argument('--top_p', type=float, default=1, required=False, help='top p')
    parser.add_argument('--top_k', type=int, default=0, required=False, help='top k')
    parser.add_argument('--seq_length', type=int, default=32, required=False, help='seq length')
    parser.add_argument('--image_path', type=str, default=None, required=False, help='image path')
    args_ = parser.parse_args()
    print(args_)
    main(args_)
