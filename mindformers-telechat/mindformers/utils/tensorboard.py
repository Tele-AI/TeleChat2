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
"""TENSORBOARD Module"""
from tensorboardX import SummaryWriter
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig

_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TENSORBOARD_WRITER_ARGS = {'log_timers_to_tensorboard': False,
                                   'log_loss_scale_to_tensorboard': False
                                   }
_GLOBAL_TENSORBOARD_TEXT_MAPPING = set()


# pylint: disable=W0613
def get_text_mapping():
    return _GLOBAL_TENSORBOARD_TEXT_MAPPING


def _default_handle(key, value, _, writer):
    global _GLOBAL_TENSORBOARD_TEXT_MAPPING
    writer.add_text(key, str(value), global_step=1)
    _GLOBAL_TENSORBOARD_TEXT_MAPPING.add(key)


def _handle_context(key, value, config, writer):
    """Handle context"""
    value.update({'device_id': config.rank_id})
    _default_handle(key, value, config, writer)


def _handle_model(key, value, config, writer):
    """Handle context"""
    value['model_config'].pop('checkpoint_name_or_path', 'base_model')
    _default_handle(key, value, config, writer)


def _handle_parallel(key, value, config, writer):
    """Handle parallel"""
    value.pop('device_num', 8)
    _default_handle(key, value, config, writer)


def _handle_runner_config(key, value, config, writer):
    """Handle runner config."""
    if config.runner_config.batch_size == config.runner_config.global_batch_size:
        _RUNNER_CONFIG['batch_size'] = (config.runner_config.global_batch_size //
                                        config.runner_config.gradient_accumulation_steps //
                                        config.parallel_config.data_parallel //
                                        config.micro_batch_interleave_num)
    else:
        _RUNNER_CONFIG['batch_size'] = config.runner_config.batch_size
    _RUNNER_CONFIG['epochs'] = config.runner_config.origin_epochs
    _RUNNER_CONFIG['sink_mode'] = config.runner_config.sink_mode
    _RUNNER_CONFIG['sink_size'] = config.runner_config.sink_size
    _RUNNER_CONFIG['gradient_accumulation_steps'] = config.runner_config.gradient_accumulation_steps
    _default_handle('runner_config', _RUNNER_CONFIG, config, writer)


def _handle_runner_wrapper(key, value, config, writer):
    """Write moe config."""
    if isinstance(config.runner_wrapper['scale_sense'], DynamicLossScaleUpdateCell):
        config.runner_wrapper['scale_sense'] = {
            'type': DynamicLossScaleUpdateCell,
            'loss_scale_value': config.runner_wrapper['scale_sense'].loss_scale_value,
            'scale_factor': config.runner_wrapper['scale_sense'].scale_factor,
            'scale_window': config.runner_wrapper['scale_sense'].scale_window,
        }
    else:
        logger.info("The scale_sense of the runner_wrapper will not be logged specifically.")
    _default_handle('runner_wrapper', config.runner_wrapper, config, writer)


def _handle_moe_config(_, value, config, writer):
    """Handle moe config."""
    cfg_dict = value.to_dict()
    _default_handle('moe_config', cfg_dict, config, writer)


def _handle_parallel_config(_, value, config, writer):
    """Handle parallel config."""
    cfg_dict = value.to_dict()
    _default_handle('parallel_config', cfg_dict, config, writer)


def _handle_recompute_config(_, value, config, writer):
    """Handle parallel config."""
    cfg_dict = value.to_dict()
    _default_handle('recompute_config', cfg_dict, config, writer)


def _handle_tensorboard(key, value, config, writer):
    """Handle parallel config."""
    queue_size = _check_queue_size(config.tensorboard.tensorboard_queue_size)
    value.update({'tensorboard_queue_size': queue_size})
    for key_, value_ in _GLOBAL_TENSORBOARD_WRITER_ARGS.items():
        value.update({key_: value_})
    _default_handle(key, value, config, writer)

handlers = {
    'context': _handle_context,
    'model': _handle_model,
    'parallel': _handle_parallel,
    'runner_config': _handle_runner_config,
    'runner_wrapper': _handle_runner_wrapper,
    'moe_config': _handle_moe_config,
    'parallel_config': _handle_parallel_config,
    'recompute_config': _handle_recompute_config,
    'tensorboard': _handle_tensorboard
}

conditions = {
    'ignore_data_skip': [('resume_training', False)],
    'data_skip_steps': [('resume_training', False), ('ignore_data_skip', True)],
    'load_ckpt_format': [('load_checkpoint', None)],
    'auto_trans_ckpt': [('load_checkpoint', None)],
    'transform_process_num': [('load_checkpoint', None), ('auto_trans_ckpt', False)],
    'src_strategy_path_or_dir': [('load_checkpoint', None), ('auto_trans_ckpt', False)],
    'load_ckpt_async': [('load_checkpoint', None)],
    'profile_communication': [('profile', False)],
    'profile_level': [('profile', False)],
    'profile_memory': [('profile', False)],
    'profile_start_step': [('profile', False)],
    'profile_stop_step': [('profile', False)],
    'profile_rank_ids': [('profile', False)],
    'profile_pipeline': [('profile', False)],
    'init_start_profile': [('profile', False)],
    'lr_scale_factor': [('lr_scale', False)],
    'eval_callbacks': [('do_eval', False)],
    'eval_step_interval': [('do_eval', False)],
    'eval_epoch_interval': [('do_eval', False)],
    'eval_dataset': [('do_eval', False)],
    'eval_dataset_task': [('do_eval', False)],
}

_TENSORBOARD_TEXT_UNWRITTEN_MAPPING = {
    'adapter_id',
    'enable_mindio_ttp_save_ckpt',
    'exclude_cann_cpu',
    'infer_precision_sync',
    'local_rank',
    'rank_id',
    'postprocess_use_numpy',
    'train_precision_sync',
    'processor',
    'input_data',
    'predict_batch_size',
    'auto_tune',
    'autotune_per_step',
    'save_file',
    'filepath_prefix'
}

_RUNNER_CONFIG = {
    'epochs': 1,
    'batch_size': 1,
    'sink_mode': False,
    'sink_size': 1,
    'gradient_accumulation_steps': 1
}


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def _set_tensorboard_writer(config):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER

    if hasattr(config, 'tensorboard_dir') and config.tensorboard_dir:
        if _GLOBAL_TENSORBOARD_WRITER is not None:
            logger.warning("tensorboard writer is already initialized.")
        else:
            logger.info('......setting tensorboard......')
            queue_size = getattr(config, 'tensorboard_queue_size', 10)
            queue_size = _check_queue_size(queue_size)
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=config.tensorboard_dir,
                max_queue=queue_size)


def _unset_tensorboard_writer():
    """Unset tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    if _GLOBAL_TENSORBOARD_WRITER is not None:
        if isinstance(_GLOBAL_TENSORBOARD_WRITER, SummaryWriter):
            _GLOBAL_TENSORBOARD_WRITER.close()
        else:
            logger.warning("The global tensorboard writer is not an instance of SummaryWriter.")
        _GLOBAL_TENSORBOARD_WRITER = None


def write_args_to_tensorboard(config):
    """Write arguments to tensorboard."""
    global _GLOBAL_TENSORBOARD_WRITER
    if _GLOBAL_TENSORBOARD_WRITER is None:
        raise ValueError("tensorboard writer is not set.")
    writer = get_tensorboard_writer()
    if isinstance(config, MindFormerConfig):
        for key, value in config.items():
            if key in _TENSORBOARD_TEXT_UNWRITTEN_MAPPING:
                continue
            elif key in conditions:
                if any(config.get_value(attr, flag) == flag for (attr, flag) in conditions[key]):
                    continue

            handler = handlers.get(key)

            if handler:
                handler(key, value, config, writer)
            else:
                _default_handle(key, value, config, writer)
    else:
        logger.warning("Only MindFormerConfig will be written.")


def update_tensorboard_args(config):
    """Update args for tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER_ARGS
    for key in _GLOBAL_TENSORBOARD_WRITER_ARGS:
        if isinstance(config, MindFormerConfig):
            if config.get_value(key) is not None:
                _GLOBAL_TENSORBOARD_WRITER_ARGS[key] = config.get_value(key)
        else:
            if hasattr(config, key) and getattr(config, key) is not None:
                _GLOBAL_TENSORBOARD_WRITER_ARGS[key] = getattr(config, key)


def get_tensorboard_args():
    """Return args for tensorboard writer."""
    return _GLOBAL_TENSORBOARD_WRITER_ARGS


def _check_queue_size(queue_size):
    """Check if queue_size is a positive int"""
    if queue_size is None:
        return 10
    if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size <= 0:
        logger.warning(f"tensorboard.queue_size will be set to 10. Since it needs to be a positive integer, "
                       f"but got {queue_size}.")
        return 10
    return queue_size
