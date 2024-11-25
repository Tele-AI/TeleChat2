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
"""LOGGER Module"""
import logging
import logging.config
import logging.handlers
import os
import stat
import sys
import traceback

from functools import wraps
from tempfile import TemporaryFile
from typing import Dict, List, Tuple, Union

from mindformers.tools.utils import get_num_nodes_devices, get_rank_info, \
    convert_nodes_devices_input, generate_rank_list,\
    check_in_modelarts, check_list, LOCAL_DEFAULT_PATH,\
    get_log_path


logger_list = []
LEVEL = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
MODELARTS_LOG_FILE_DIR = '/cache/ma-user-work/'
LOCAL_DEFAULT_LOG_FILE_DIR = os.path.join(
    os.getenv("LOCAL_DEFAULT_PATH", LOCAL_DEFAULT_PATH), 'log')
RANK_DIR_FORMATTER = 'rank_{}/'
DEFAULT_FILEHANDLER_FORMAT = '[%(levelname)s] %(asctime)s ' \
                             '[%(filepath)s:%(lineno)d] %(funcName)s: %(message)s'
DEFAULT_STDOUT_FORMAT = '%(asctime)s - %(name)s[%(filepath)s:%(lineno)d] - %(levelname)s - %(message)s'
DEFAULT_REDIRECT_FILE_NAME = 'mindspore.log'


class _DataFormatter(logging.Formatter):
    """Log formatter"""
    def format(self, record):
        """
        Apply log format with specified pattern.

        Args:
            record (str): Format pattern.

        Returns:
            str, formatted log content according to format pattern.
        """
        # NOTICE: when the Installation directory of mindspore changed,
        # ms_home_path must be changed
        ms_install_home_path = 'mindformers'
        idx = record.pathname.rfind(ms_install_home_path)
        if idx >= 0:
            # Get the relative path of the file
            record.filepath = record.pathname[idx:]
        else:
            record.filepath = record.pathname
        return super().format(record)


def _get_stack_info(frame):
    """
    Get the stack information.

    Args:
        frame(frame): the frame requiring information.

    Returns:
        str, the string of the stack information.
    """
    stack_prefix = 'Stack (most recent call last):\n'
    sinfo = stack_prefix + "".join(traceback.format_stack(frame))
    return sinfo


# pylint: disable=W0613
def _find_caller(stack_info=False, stacklevel=1):
    """
    Find the stack frame of the caller.

    Override findCaller on the logger, Support for getting log record.
    Find the stack frame of the caller so that we can note the source
    file name, function name and line number.

    Args:
        stack_info (bool): If the value is true, print stack information to the log. Default: False.

    Returns:
        tuple, the tuple of the frame data.
    """
    # pylint: disable=W0212
    f = sys._getframe(3)
    sinfo = None
    # log_file is used to check caller stack frame
    log_file = os.path.normcase(f.f_code.co_filename)
    f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)", None
    while f:
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if log_file == filename:
            f = f.f_back
            continue
        if stack_info:
            sinfo = _get_stack_info(f)
        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
        break
    return rv


def judge_redirect(rank_id: int,
                   rank_size: int,
                   redirect_nodes: Union[List, Tuple, Dict[str, int], None] = None,
                   redirect_devices: Union[List, Tuple, Dict[str, int], None] = None):
    """Determine if the stderr of this process needs to be redirected.

    Args:
        rank_id (int): Rank id.
        rank_size (int): Rank Size.
        redirect_nodes (list or tuple or dict or None): Node list. The
            nodes in the list will redirect stderr.
        redirect_devices (list or tuple or dict or None): Device
            list. The devices in the list will redirect stderr.

    Returns:
        prerequisite (bool): If true, stderr will redirect.
    """
    is_redirect = True
    if rank_size > 1 and (redirect_nodes is not None or redirect_devices is not None):
        num_nodes, num_devices = get_num_nodes_devices(rank_size)
        redirect_nodes = convert_nodes_devices_input(redirect_nodes, num_nodes)
        redirect_devices = convert_nodes_devices_input(redirect_devices, num_devices)
        check_list('nodes', redirect_nodes, num_nodes)
        check_list('devices', redirect_devices, num_devices)
        rank_list = generate_rank_list(redirect_nodes, redirect_devices)
        if rank_id not in rank_list:
            is_redirect = False

    return is_redirect


class LimitedRepeatHandler(logging.StreamHandler):
    """Limited Repeat Handler"""
    def __init__(self, max_repeats=10, stream=None):
        """
        Limited Repeat Handler init

        Args:
            max_repeats(int): max repeats of same log, default: 10.
            stream: stream.
        """
        super().__init__(stream=stream)
        self.max_repeats = max_repeats
        self.latest_log = ''
        self.count = 1

    def emit(self, record):
        """emit"""
        log_message = record.getMessage()
        if log_message == self.latest_log:
            self.count += 1
            if self.count <= self.max_repeats:
                self._emit(record)
        else:
            self.count = 1
            self.latest_log = log_message
            self._emit(record)

    def _emit(self, record):
        super().emit(record)


class LimitedRepeatFileHandler(logging.handlers.RotatingFileHandler):
    """Limited Repeat File Handler"""
    def __init__(self, max_repeats=10, **kwargs):
        super().__init__(**kwargs)
        self.max_repeats = max_repeats
        self.latest_log = ''
        self.count = 1

    def emit(self, record):
        """emit"""
        log_message = record.getMessage()
        if log_message == self.latest_log:
            self.count += 1
            if self.count <= self.max_repeats:
                self._emit(record)
        else:
            self.count = 1
            self.latest_log = log_message
            self._emit(record)

    def _emit(self, record):
        super().emit(record)


class StreamRedirector:
    """Stream Re-director for Log."""

    def __init__(self, source_stream, target_stream):
        """Redirects the source stream to the target stream.

        Args:
            source_stream: Source stream.
            target_stream: Target stream.
        """
        super(StreamRedirector, self).__init__()

        self.source_stream = source_stream
        self.target_stream = target_stream

        self.save_source_stream_fd = os.dup(self.source_stream.fileno())

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            func(*args, **kwargs)
            self.stop()

        return wrapper

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """start."""
        self.source_stream.flush()
        os.dup2(self.target_stream.fileno(), self.source_stream.fileno())

    def stop(self):
        """stop."""
        self.source_stream.flush()
        os.dup2(self.save_source_stream_fd, self.source_stream.fileno())
        self.target_stream.flush()


class AiLogFastStreamRedirect2File(StreamRedirector):
    """AiLogFastStreamRedirect2File."""
    def __init__(self,
                 source_stream=None,
                 redirect_nodes: Union[List, Tuple, Dict[str, int], None] = None,
                 redirect_devices: Union[List, Tuple, Dict[str, int], None] = None,
                 **kwargs):
        """Redirect stream to file.

        Args:
            source_stream (file object or None): Streams that need to be redirected.
                Default: None, select stderr.
            redirect_nodes (list[int] or tuple[int] or optional): The computation
                nodes that will redirect stderr.
                Default is None: indicates that all nodes will redirect stderr.
                Eg [0, 1, 2, 3] or (0, 1, 2, 3): indicates that nodes 0, 1, 2,
                    and 3 all redirect stderr.
            redirect_devices (list[int] or tuple[int] or optional): The computation
                devices that will redirect stderr.
                Default is None, indicates that all devices will redirect stderr.
                Eg [0, 1, 2, 3] or (0, 1, 2, 3): indicates that devices 0, 1, 2,
                    and 3 all redirect stderr.
            kwargs (dict): File-related parameters.
                file_save_dir (str): The folder where the files that
                    save redirected stream are saved.
                append_rank_dir (bool): Whether to add a folder with the format rank{}.
                file_name (str): Redirect file name.
        """
        rank_id, rank_size = get_rank_info()

        self.is_redirect = judge_redirect(rank_id=rank_id,
                                          rank_size=rank_size,
                                          redirect_nodes=redirect_nodes,
                                          redirect_devices=redirect_devices)

        file_save_dir = kwargs.get('file_save_dir', '')
        append_rank_dir = kwargs.get('append_rank_dir', True)
        file_name = kwargs.get('file_name', '')

        if not file_save_dir:
            file_save_dir = get_log_path()
        if append_rank_dir:
            rank_str = RANK_DIR_FORMATTER.format(rank_id)
            file_save_dir = os.path.join(file_save_dir, rank_str)

        if not file_name:
            file_name = DEFAULT_REDIRECT_FILE_NAME
        self.file_path = os.path.join(file_save_dir, file_name)
        self.file_save_dir = os.path.dirname(self.file_path)

        if source_stream is None:
            source_stream = sys.stderr
        target_stream = TemporaryFile(mode='w+')

        super(AiLogFastStreamRedirect2File, self).__init__(source_stream=source_stream, target_stream=target_stream)

    def start(self):
        if self.is_redirect:
            super(AiLogFastStreamRedirect2File, self).start()

    def stop(self):
        if self.is_redirect:
            self.target_stream.flush()
            if not os.path.exists(self.file_save_dir):
                os.makedirs(self.file_save_dir, exist_ok=True)
            self.target_stream.seek(0, 0)
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.file_path, flags, modes), 'w') as fp:
                for line in self.target_stream:
                    fp.write(line)
            super(AiLogFastStreamRedirect2File, self).stop()
            self.target_stream.close()


def judge_stdout(rank_id: int,
                 rank_size: int,
                 is_output: bool,
                 nodes: Union[List, Tuple, Dict[str, int], None] = None,
                 devices: Union[List, Tuple, Dict[str, int], None] = None) -> bool:
    """Determines if logs will be output.

    Args:
        rank_id (int): Rank id.
        rank_size (int): Rank size.
        is_output (int): If set to true, logs or others will be output.
        nodes (list or tuple or dict or None): Node list. The nodes in the list
            will output the log to stdout.
        devices (list or tuple or dict or None): Device list. The devices
            in the list or output the log to stdout.

    Returns:
        is_output (bool): If set to true, logs or others will be output
            or redirect.
    """
    if is_output and rank_size > 1 and (nodes is not None or devices is not None):
        num_nodes, num_devices = get_num_nodes_devices(rank_size)
        stdout_nodes = convert_nodes_devices_input(nodes, num_nodes)
        stdout_devices = convert_nodes_devices_input(devices, num_devices)
        check_list('nodes', stdout_nodes, num_nodes)
        check_list('devices', stdout_devices, num_devices)
        rank_list = generate_rank_list(stdout_nodes, stdout_devices)
        if rank_id not in rank_list:
            is_output = False

    return is_output


def validate_nodes_devices_input(var_name: str, var):
    """Check the list of nodes or devices.

    Args:
        var_name (str): Variable name.
        var: The name of the variable to be checked.

    Returns:
        None
    """
    if not (var is None or isinstance(var, (list, tuple, dict))):
        raise TypeError('The value of {} can be None or a value of type tuple, ' 'list, or dict.'.format(var_name))
    if isinstance(var, (list, tuple)):
        for item in var:
            if not isinstance(item, int):
                raise TypeError('The elements of a variable of type list or ' 'tuple must be of type int.')


def validate_level(var_name: str, var):
    """Verify that the log level is correct.

    Args:
        var_name (str): Variable name.
        var: The name of variable to be checked.

    Returns:
        None
    """
    if not isinstance(var, str):
        raise TypeError('The format of {} must be of type str.'.format(var_name))
    if var not in LEVEL:
        raise ValueError('{}={} needs to be in {}'.format(var_name, var, LEVEL))


def validate_std_input_format(to_std: bool, stdout_nodes: Union[List, Tuple, None],
                              stdout_devices: Union[List, Tuple, None], stdout_level: str):
    """Validate the input about stdout of the get_logger function."""

    if not isinstance(to_std, bool):
        raise TypeError('The format of the to_std must be of type bool.')

    validate_nodes_devices_input('stdout_nodes', stdout_nodes)
    validate_nodes_devices_input('stdout_devices', stdout_devices)
    validate_level('stdout_level', stdout_level)


def validate_file_input_format(file_level: Union[List[str], Tuple[str]], file_save_dir: str, append_rank_dir: str,
                               file_name: Union[List[str], Tuple[str]]):
    """Validate the input about file of the get_logger function."""

    if not isinstance(file_level, (tuple, list)):
        raise TypeError('The value of file_level should be list or a tuple.')
    for level in file_level:
        validate_level('level in file_level', level)

    if not len(file_level) == len(file_name):
        raise ValueError('The length of file_level and file_name should be equal.')

    if not isinstance(file_save_dir, str):
        raise TypeError('The value of file_save_dir should be a value of type str.')

    if not isinstance(append_rank_dir, bool):
        raise TypeError('The value of append_rank_dir should be a value of type bool.')

    if not isinstance(file_name, (tuple, list)):
        raise TypeError('The value of file_name should be list or a tuple.')
    for name in file_name:
        if not isinstance(name, str):
            raise TypeError('The value of name in file_name should be a value of type str.')


def _convert_level(level: str) -> int:
    """Convert the format of the log to logging level.

    Args:
        level (str): User log level.

    Returns:
        level (str): Logging level.
    """
    level_convert = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_convert.get(level, logging.INFO)

    return level


def get_logger(logger_name: str = 'mindformers', **kwargs) -> logging.Logger:
    """Get the logger. Both computing centers and bare metal servers are
    available.

    Args:
        logger_name (str): Logger name.
        kwargs (dict): Other input.
            to_std (bool): If set to True, output the log to stdout.
            stdout_nodes (list[int] or tuple[int] or optional):
                The computation nodes that will output the log to stdout.
                default: None, indicates that all nodes will output logs to stdout.
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): indicates that nodes 0, 1, 2, and
                    3 all output logs to stdout.
            stdout_devices (list[int] or tuple[int] or optional):
                The computation devices that will output the log to stdout.
                default: None, indicates that all devices will output logs to stdout.
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): indicates that devices 0, 1, 2,
                    and 3 all output logs to stdout.
            stdout_level (str): The level of the log output to stdout.
                If the type is str, the options are DEBUG, INFO, WARNING, ERROR, CRITICAL.
            stdout_format (str): Log format.
            file_level (list[str] or tuple[str]): The level of the log output to file.
                eg: ['INFO', 'ERROR'] Indicates that the logger will output logs above
                    the level INFO and ERROR in the list to the corresponding file.
                The length of the list needs to be the same as the length of file_name.
            file_save_dir (str): The folder where the log files are stored.
            append_rank_dir (bool): Whether to add a folder with the format rank{}.
            file_name (list[str] or list[tuple]): Store a list of output file names.
            max_file_size (int): The maximum size of a single log file. Unit: MB.
            max_num_of_files (int): The maximum number of files to save.

    Returns:
        logger (logging.Logger): Logger.
    """
    mf_logger = logging.getLogger(logger_name)
    mf_logger.findCaller = _find_caller
    if logger_name in logger_list:
        return mf_logger

    to_std = kwargs.get('to_std', True)
    stdout_nodes = kwargs.get('stdout_nodes', None)

    rank_id, rank_size = get_rank_info()

    def get_stdout_devices():
        if os.getenv("STDOUT_DEVICES"):
            devices = os.getenv("STDOUT_DEVICES")
            if devices.startswith(("(", "[")) and devices.endswith((")", "]")):
                devices = devices[1:-1]
            devices = tuple(map(lambda x: int(x.strip()), devices.split(",")))
        elif check_in_modelarts():
            devices = kwargs.get('stdout_devices', (min(rank_size - 1, 7),))
        else:
            devices = kwargs.get('stdout_devices', None)
        return devices

    stdout_devices = get_stdout_devices()
    stdout_level = kwargs.get('stdout_level', 'INFO')
    stdout_format = kwargs.get('stdout_format', '')
    file_level = kwargs.get('file_level', ('INFO', 'ERROR'))
    file_save_dir = kwargs.get('file_save_dir', '')
    append_rank_dir = kwargs.get('append_rank_dir', True)
    file_name = kwargs.get('file_name', (f'info.log', 'error.log'))
    max_file_size = kwargs.get('max_file_size', 50)
    max_num_of_files = kwargs.get('max_num_of_files', 5)

    validate_std_input_format(to_std, stdout_nodes, stdout_devices, stdout_level)
    validate_file_input_format(file_level, file_save_dir, append_rank_dir, file_name)

    to_std = judge_stdout(rank_id=rank_id,
                          rank_size=rank_size,
                          is_output=to_std,
                          nodes=stdout_nodes,
                          devices=stdout_devices)
    if to_std:
        if not stdout_format:
            stdout_format = DEFAULT_STDOUT_FORMAT
        stream_handler = LimitedRepeatHandler(10, sys.stdout)
        stream_handler.setLevel(_convert_level(stdout_level))
        stream_formatter = _DataFormatter(stdout_format)
        stream_handler.setFormatter(stream_formatter)
        mf_logger.addHandler(stream_handler)

    logging_level = []
    for level in file_level:
        logging_level.append(_convert_level(level))

    if not file_save_dir:
        file_save_dir = get_log_path()
    rank_dir = RANK_DIR_FORMATTER
    if append_rank_dir:
        rank_str = rank_dir.format(rank_id)
        file_save_dir = os.path.join(file_save_dir, rank_str)

    file_path = []
    for name in file_name:
        path = os.path.join(file_save_dir, name)
        path = os.path.realpath(path)
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        file_path.append(path)

    max_file_size = max_file_size * 1024 * 1024

    file_formatter = _DataFormatter(DEFAULT_FILEHANDLER_FORMAT)
    for i, level in enumerate(file_level):
        file_handler = LimitedRepeatFileHandler(filename=file_path[i],
                                                maxBytes=max_file_size,
                                                backupCount=max_num_of_files,
                                                encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        mf_logger.addHandler(file_handler)

    mf_logger.setLevel(_convert_level('INFO'))

    mf_logger.propagate = False

    logger_list.append(logger_name)

    return mf_logger


class _LogActionOnce:
    """
    A wrapper for modify the warning logging to an empty function. This is used when we want to only log
    once to avoid the repeated logging.

    Args:
        logger (logging): The logger object.

    """
    is_logged = dict()

    def __init__(self, m_logger, key, no_warning=False):
        self.logger = m_logger
        self.key = key
        self.no_warning = no_warning

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not hasattr(self.logger, 'warning'):
                return func(*args, **kwargs)

            old_func = self.logger.warning
            if self.no_warning or self.key in _LogActionOnce.is_logged:
                self.logger.warning = lambda x: x
            else:
                _LogActionOnce.is_logged[self.key] = True
            res = func(*args, **kwargs)
            if hasattr(self.logger, 'warning'):
                self.logger.warning = old_func
            return res

        return wrapper


# pylint: disable=C0103
class logger:
    """A class to call logger"""
    @classmethod
    def info(cls, msg, *args, **kwargs):
        """Log a message with severity 'INFO' on the Mindformers logger."""
        get_logger().info(msg, *args, **kwargs)

    @classmethod
    def debug(cls, msg, *args, **kwargs):
        """Log a message with severity 'DEBUG' on the Mindformers logger."""
        get_logger().debug(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        """Log a message with severity 'ERROR' on the Mindformers logger."""
        get_logger().error(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg, *args, **kwargs):
        """Log a message with severity 'WARNING' on the Mindformers logger."""
        get_logger().warning(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg, *args, **kwargs):
        """Log a message with severity 'CRITICAL' on the Mindformers logger."""
        get_logger().critical(msg, *args, **kwargs)
