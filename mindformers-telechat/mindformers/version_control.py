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
"""MindSpore Version Control"""
import os
import mindspore as ms
from mindspore import nn, mint
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.ops.auto_generate import Scatter  # internal api for aclnn op

from mindformers.tools.utils import get_predict_run_mode
from .tools.utils import is_version_ge
from .tools.logger import logger


def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_ge(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext
        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    if ascend_chip_type == "UNSET":
        logger.info("Environment variables need to be set manually to obtain the chip type,"
                    "which can be set as follows: \n"
                    "For Atlas 800, run 'export ASCEND_CHIP_TYPE=910a' before the program runs.\n"
                    "For Atlas 800T A2, run 'export ASCEND_CHIP_TYPE=910b' before the program runs.\n"
                    "If you need to get chip information automatically, MindSpore 2.2 and above is recommended")
    return ascend_chip_type


def is_910a():
    device = get_ascend_soc_version()
    return device in ['910a', 'ascend910']


def is_910b():
    device = get_ascend_soc_version()
    return device in ['910b', 'ascend910b']


def get_predict_lazy_inline(func):
    """Predict lazy inline decorator."""

    def decorator(*args, **kwargs):
        if get_predict_run_mode():
            from mindspore.common import lazy_inline
            lazy_inline(func)(*args, **kwargs)

            logger.info("Predict enable lazy inline.")
        else:
            func(*args, **kwargs)

    return decorator


def check_lazy_inline_version():
    if not is_version_ge(ms.__version__, "2.2.0"):
        logger.info("The Lazy Inline compilation acceleration feature is not supported "
                    "when MindSpore version is earlier than 2.2.0, The current MindSpore version is %s, "
                    "please install MindSpore 2.2.0 or later.", ms.__version__)
        return False
    return True


def get_lazy_inline(func):
    """Lazy inline decorator."""

    def decorator(*args, **kwargs):
        stand_alone = ms.get_auto_parallel_context("parallel_mode") == 'stand_alone'
        pipline_parallel = ms.get_auto_parallel_context("pipeline_stages") > 1

        if os.getenv("ENABLE_LAZY_INLINE", "1") == "0":
            logger.info("The Lazy Inline compilation acceleration feature is turned off, due to the "
                        "environment variable ENABLE_LAZY_INLINE is set to 0.")
            func(*args, **kwargs)
            return

        if not check_lazy_inline_version():
            func(*args, **kwargs)
            return

        if stand_alone:
            logger.info("The Lazy Inline compilation acceleration feature does not support single-card mode."
                        "This feature is disabled by default. ENABLE_LAZY_INLINE=1 does not take effect.")
            func(*args, **kwargs)
            return

        if not pipline_parallel and os.getenv("ENABLE_LAZY_INLINE_NO_PIPELINE", "0") == "0":
            logger.info("The Lazy Inline compilation acceleration feature "
                        "only works in pipeline parallel mode (pipeline_stage > 1). "
                        "Current pipeline stage=1, the feature is disabled by default. "
                        "You can also enable lazy inline without pipeline parallel, by setting "
                        "environment variable `export ENABLE_LAZY_INLINE_NO_PIPELINE=1`.")
            func(*args, **kwargs)
            return

        from mindspore.common import lazy_inline
        lazy_inline(func)(*args, **kwargs)

    return decorator


def get_dropout(dropout_prob):
    if is_version_ge(ms.__version__, '1.11.0'):
        dropout = nn.Dropout(p=dropout_prob)
    else:
        dropout = nn.Dropout(keep_prob=1 - dropout_prob)
    return dropout


def get_tril():
    if is_version_ge(ms.__version__, '1.11.0'):
        tril = P.Tril()
    else:
        tril = nn.Tril()
    return tril


def get_norm():
    """return ops.norm"""

    # pylint: disable=C0103
    # pylint: disable=W0622
    def tensor_norm1(A, ord=None, dim=None, keepdim=False, dtype=None):
        return F.norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

    # pylint: disable=C0103
    # pylint: disable=W0622
    def tensor_norm2(A, ord=2, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            logger.warning("The 'dtype' is not available when mindspore version < '1.11.0'")
        if not isinstance(ord, int):
            raise TypeError("The type of 'ord' should be int when mindspore version < '1.11.0'")
        return F.norm(A, dim, p=ord, keep_dims=keepdim)

    if is_version_ge(ms.__version__, '1.11.0'):
        return tensor_norm1
    return tensor_norm2


def get_dataset_map(dataset, operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs):
    if is_version_ge(ms.__version__, "1.11.0"):
        return dataset.map(operations,
                           input_columns=input_columns,
                           output_columns=output_columns,
                           num_parallel_workers=num_parallel_workers,
                           **kwargs)
    return dataset.map(operations,
                       input_columns=input_columns,
                       output_columns=output_columns,
                       column_order=output_columns,
                       num_parallel_workers=num_parallel_workers,
                       python_multiprocessing=kwargs.get('python_multiprocessing', False),
                       max_rowsize=kwargs.get('max_rowsize', 16),
                       cache=kwargs.get('cache', None),
                       callbacks=kwargs.get('callbacks', None),
                       offload=kwargs.get('offload', None))


def get_identity():
    if is_version_ge(ms.__version__, "1.11.0"):
        return nn.Identity()
    return F.identity


def fix_optim_global_step_sig():
    # when the version of mindspore bigger than 2.2.0, it should update global step explicitly.
    return is_version_ge(ms.__version__, "2.2.0")


def check_valid_flash_attention(import_fa_valid=True, fa_type=None):
    """check mindspore version is valid for input flash attention"""
    version_map = {"PromptFlashAttention": "2.2.0",
                   "FlashAttention": "2.2.0"}
    valid_version = version_map.get(fa_type)
    if not is_910b() and fa_type in ["PromptFlashAttention"]:
        logger.warning(f"Current device {get_ascend_soc_version()} do not support {fa_type}, "
                       f"please use 910b device.")
        return False
    if valid_version is None:
        raise ValueError(f"fa_type should be in {list(version_map.keys())}, but get {fa_type}")
    version_valid = is_version_ge(ms.__version__, valid_version)
    if not version_valid:
        logger.warning(f"Current MindSpore do not support {fa_type}, "
                       f"please upgrade to {valid_version} or later version.")
        logger.warning("Now running on self-attention mode.")
        result = False
    elif not import_fa_valid:
        logger.warning(f"Import {fa_type} ERROR, please upgrade your MindSpore to {valid_version} or later version. ")
        logger.warning("Now running on self-attention mode.")
        result = False
    # both pass should return True
    else:
        result = True
    return result


def choose_flash_attention_dtype():
    """
    attention_mask dtype should be float16 on ms 2.2.0, uint8 on 2.2.10
    ms version below 2.2.0 won't be in this func
    """
    fa_dtype = ms.uint8
    cur_ver = ms.__version__
    if is_version_ge(cur_ver, "2.2.0") and not is_version_ge(cur_ver, "2.2.1"):
        fa_dtype = ms.float16
    elif is_version_ge(cur_ver, "2.2.1"):
        fa_dtype = ms.uint8
    return fa_dtype


def check_valid_big_kernel():
    """check mindspore version is valid for big kernel SiLU and LlamaRMSNorm Ops"""
    version_valid = is_version_ge(ms.__version__, "2.2.10")
    # below ms 2.2.10 is not support
    if not version_valid:
        logger.warning("Current MindSpore do not support big kernel SiLU and RMSNorm, "
                       "please upgrade to 2.2.10 or later version.")
        result = False
    else:
        result = True
    return result


def is_version_python(cur_ver, tar_ver):
    """
        return cur_ver >= tar_ver.
        Check whether the current version is higher than or equal to the base version.
        for cur_ver: 3.7.10, tar_ver: 3.9.0, it return False.
        you can get python cur_ver through:
            cur_py_ver = sys.version.split(' ')[0]
    """
    version_split_char = '.'
    if version_split_char not in tar_ver or version_split_char not in cur_ver:
        raise ValueError("The version string will contain the `.`."
                         "For example, cur_ver: 3.7.10， tar_ver: 3.9.0")
    for x, y in zip(cur_ver.split(version_split_char), tar_ver.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def check_rmsnorm_big_kernel_valid():
    """check whether rmsnorm big kernel is valid"""
    if check_valid_big_kernel() and not is_910a():
        return True
    return False


def use_mint_op():
    """Check whether ms.mint op is valid or not"""
    version_info = ms.__version__.split('rc')
    return is_version_ge(version_info[0], '2.3.0') and ms.__version__ != '2.3.0rc1'


def check_valid_gmm_op():
    """check mindspore version is valid for groupedmatmul"""
    version_valid = is_version_ge(ms.__version__, "2.3.0")
    if not version_valid:
        logger.warning(f"Current MindSpore do not support GroupedMatmul, "
                       f"please upgrade to {version_valid} or later version.")
        return False
    return True


def check_valid_moefinalizerouting_op():
    """check mindspore version is valid for groupedmatmul"""
    version_valid = is_version_ge(ms.__version__, "2.3.0")
    if not version_valid:
        logger.warning(f"Current MindSpore do not support MoeFinalizeRouting, "
                       f"please upgrade to {version_valid} or later version.")
        return False
    return True


def check_valid_mindspore_gs():
    """check mindspore golden-stick version is valid or not"""
    import mindspore_gs
    version_valid = is_version_ge(mindspore_gs.__version__, "0.6.0")
    if not version_valid:
        logger.warning(f"Current MindSpore Gloden-Stick version does not match"
                       f"the MindFormers version, please upgrade to {version_valid} or later version.")
        return False
    return True


def get_scatter():
    """
    Return:
        `Scatter()` when MindSpore Version is less than 2.4.0.
        `mint.scatter` when MindSpore Version is 2.4.0 or later.
    """
    if is_version_ge(ms.__version__, "2.4.0"):
        return mint.scatter
    return Scatter()
