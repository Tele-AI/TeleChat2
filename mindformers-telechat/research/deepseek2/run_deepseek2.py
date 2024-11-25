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
"""Run deepseekv2."""
import argparse
import os

from mindformers.tools.register import MindFormerConfig, ActionDict
from mindformers.tools.utils import str2bool, parse_value
from mindformers.core.context import build_context
from mindformers.trainer import Trainer
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.tools.logger import logger
from mindformers.tools import set_output_path

# pylint: disable=W0611
import deepseek2
import deepseek2_config


@cloud_monitor()
def main(config):
    """main."""
    # set output path
    set_output_path(config.output_dir)

    # init context
    build_context(config)

    trainer = Trainer(config)
    if config.run_mode == 'train' or config.run_mode == 'finetune':
        trainer.train()
    elif config.run_mode == 'eval':
        trainer.evaluate(eval_checkpoint=config.load_checkpoint)
    elif config.run_mode == 'predict':
        trainer.predict(predict_checkpoint=config.load_checkpoint, input_data=config.input_data,
                        batch_size=config.predict_batch_size)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="configs/mae/run_mae_vit_base_p16_224_800ep.yaml",
        required=True,
        help='YAML config files')
    parser.add_argument(
        '--mode', default=None, type=int,
        help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: GRAPH_MODE(0).'
             'GRAPH_MODE or PYNATIVE_MODE can be set by `mode` attribute and both modes support all backends,'
             'Default: None')
    parser.add_argument(
        '--device_id', default=None, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1], '
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--device_target', default=None, type=str,
        help='The target device to run, support "Ascend", "GPU", and "CPU".'
             'If device target is not set, the version of MindSpore package is used.'
             'Default: None')
    parser.add_argument(
        '--run_mode', default=None, type=str,
        help='task running status, it support [train, finetune, eval, predict].'
             'Default: None')
    parser.add_argument(
        '--do_eval', default=None, type=str2bool,
        help='whether do evaluate in training process.'
             'Default: None')
    parser.add_argument(
        '--train_dataset_dir', default=None, type=str,
        help='dataset directory of data loader to train/finetune. '
             'Default: None')
    parser.add_argument(
        '--eval_dataset_dir', default=None, type=str,
        help='dataset directory of data loader to eval. '
             'Default: None')
    parser.add_argument(
        '--predict_data', default=None, type=str, nargs='+',
        help='input data for predict, it support real data path or data directory.'
             'Default: None')
    parser.add_argument(
        '--predict_batch_size', default=None, type=int,
        help='batch size for predict data, set to perform batch predict.'
             'Default: None')
    parser.add_argument(
        '--load_checkpoint', default=None, type=str,
        help="load model checkpoint to train/finetune/eval/predict, "
             "it is also support input model name, such as 'mae_vit_base_p16', "
             "please refer to https://gitee.com/mindspore/mindformers#%E4%BB%8B%E7%BB%8D."
             "Default: None")
    parser.add_argument(
        '--src_strategy_path_or_dir', default=None, type=str,
        help="The strategy of load_checkpoint, "
             "if dir, it will be merged before transform checkpoint, "
             "if file, it will be used in transform checkpoint directly, "
             "Default: None, means load_checkpoint is a single whole ckpt, not distributed")
    parser.add_argument(
        '--auto_trans_ckpt', default=None, type=str2bool,
        help="if true, auto transform load_checkpoint to load in distributed model. ")
    parser.add_argument(
        '--transform_process_num', default=None, type=int,
        help="The number of processes responsible for checkpoint transform.")
    parser.add_argument(
        '--only_save_strategy', default=None, type=str2bool,
        help="if true, when strategy files are saved, system exit. ")
    parser.add_argument(
        '--resume_training', default=None, type=str2bool,
        help="Decide whether to resume training or specify the name of the checkpoint "
             "from which to resume training.")
    parser.add_argument(
        '--strategy_load_checkpoint', default=None, type=str,
        help='path to parallel strategy checkpoint to load, it support real data path or data directory.'
             'Default: None')
    parser.add_argument(
        '--remote_save_url', default=None, type=str,
        help='remote save url, where all the output files will tansferred and stroed in here. '
             'Default: None')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='global random seed to train/finetune.'
             'Default: None')
    parser.add_argument(
        '--use_parallel', default=None, type=str2bool,
        help='whether use parallel mode. Default: None')
    parser.add_argument(
        '--profile', default=None, type=str2bool,
        help='whether use profile analysis. Default: None')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')
    parser.add_argument(
        '--epochs', default=None, type=int,
        help='train epochs.'
             'Default: None')
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='batch_size of datasets.'
             'Default: None')
    parser.add_argument(
        '--gradient_accumulation_steps', default=None, type=int,
        help='Number of updates steps to accumulate before performing a backward/update pass.'
             'Default: None')
    parser.add_argument(
        '--sink_mode', default=None, type=str2bool,
        help='whether use sink mode. '
             'Default: None')
    parser.add_argument(
        '--num_samples', default=None, type=int,
        help='number of datasets samples used.'
             'Default: None')
    parser.add_argument(
        '--output_dir', default=None, type=str,
        help='output directory.')
    parser.add_argument(
        '--register_path', default=None, type=str,
        help='the register path of outer API.')

    args_, rest_args_ = parser.parse_known_args()
    rest_args_ = [i for item in rest_args_ for i in item.split("=")]
    if len(rest_args_) % 2 != 0:
        raise ValueError(f"input arg key-values are not in pair, please check input args. ")

    if args_.config is not None and not os.path.isabs(args_.config):
        args_.config = os.path.join(work_path, args_.config)

    if args_.register_path is not None:
        if not os.path.isabs(args_.register_path):
            args_.register_path = os.path.join(work_path, args_.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = args_.register_path
    config_ = MindFormerConfig(args_.config)

    if args_.device_id is not None:
        config_.context.device_id = args_.device_id
    if args_.device_target is not None:
        config_.context.device_target = args_.device_target
    if args_.mode is not None:
        config_.context.mode = args_.mode
    if args_.run_mode is not None:
        config_.run_mode = args_.run_mode
    if args_.do_eval is not None:
        config_.do_eval = args_.do_eval
    if args_.seed is not None:
        config_.seed = args_.seed
    if args_.use_parallel is not None:
        config_.use_parallel = args_.use_parallel
    if args_.load_checkpoint is not None:
        config_.load_checkpoint = args_.load_checkpoint
    if args_.src_strategy_path_or_dir is not None:
        config_.src_strategy_path_or_dir = args_.src_strategy_path_or_dir
    if args_.auto_trans_ckpt is not None:
        config_.auto_trans_ckpt = args_.auto_trans_ckpt
    if args_.transform_process_num is not None:
        config_.transform_process_num = args_.transform_process_num
    if args_.only_save_strategy is not None:
        config_.only_save_strategy = args_.only_save_strategy
    if args_.resume_training is not None:
        config_.resume_training = args_.resume_training
    if args_.strategy_load_checkpoint is not None:
        if os.path.isdir(args_.strategy_load_checkpoint):
            ckpt_list = [os.path.join(args_.strategy_load_checkpoint, file)
                         for file in os.listdir(args_.strategy_load_checkpoint) if file.endwith(".ckpt")]
            args_.strategy_load_checkpoint = ckpt_list[0]
        config_.parallel.strategy_ckpt_load_file = args_.strategy_load_checkpoint
    if args_.remote_save_url is not None:
        config_.remote_save_url = args_.remote_save_url
    if args_.profile is not None:
        config_.profile = args_.profile
    if args_.options is not None:
        config_.merge_from_dict(args_.options)
    if config_.run_mode not in ['train', 'eval', 'predict', 'finetune']:
        raise ValueError(f"run status must be in {['train', 'eval', 'predict', 'finetune']}, "
                         f"but get {config_.run_mode}")
    if args_.train_dataset_dir:
        config_.train_dataset.data_loader.dataset_dir = args_.train_dataset_dir
    if args_.eval_dataset_dir:
        config_.eval_dataset.data_loader.dataset_dir = args_.eval_dataset_dir
    if config_.run_mode == 'predict':
        if args_.predict_data is None:
            logger.info("dataset by config is used as input_data.")
        if isinstance(args_.predict_data, list):
            if len(args_.predict_data) > 1:
                logger.info("predict data is a list, take it as input text list.")
            else:
                args_.predict_data = args_.predict_data[0]
        if isinstance(args_.predict_data, str):
            if os.path.isdir(args_.predict_data):
                predict_data = []
                for root, _, file_list in os.walk(os.path.join(args_.predict_data)):
                    for file in file_list:
                        if file.lower().endswith((".jpg", ".png", ".jpeg", ".JPEG", ".bmp")):
                            predict_data.append(os.path.join(root, file))
                args_.predict_data = predict_data
            else:
                args_.predict_data = args_.predict_data.replace(r"\n", "\n")

        config_.input_data = args_.predict_data
        if args_.predict_batch_size is not None:
            config_.predict_batch_size = args_.predict_batch_size
    if args_.epochs is not None:
        config_.runner_config.epochs = args_.epochs
    if args_.batch_size is not None:
        config_.runner_config.batch_size = args_.batch_size
    if args_.gradient_accumulation_steps is not None:
        config_.runner_config.gradient_accumulation_steps = args_.gradient_accumulation_steps
    if args_.sink_mode is not None:
        config_.runner_config.sink_mode = args_.sink_mode
    if args_.num_samples is not None:
        if config_.train_dataset and config_.train_dataset.data_loader:
            config_.train_dataset.data_loader.num_samples = args_.num_samples
        if config_.eval_dataset and config_.eval_dataset.data_loader:
            config_.eval_dataset.data_loader.num_samples = args_.num_samples
    if args_.output_dir is not None:
        config_.output_dir = args_.output_dir

    while rest_args_:
        key = rest_args_.pop(0)
        value = rest_args_.pop(0)
        if not key.startswith("--"):
            raise ValueError("Custom config key need to start with --.")
        dists = key[2:].split(".")
        dist_config = config_
        while len(dists) > 1:
            if dists[0] not in dist_config:
                raise ValueError(f"{dists[0]} is not a key of {dist_config}, please check input arg keys. ")
            dist_config = dist_config[dists.pop(0)]
        dist_config[dists.pop()] = parse_value(value)

    main(config_)
