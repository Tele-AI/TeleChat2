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
"""MindFormers Tools."""
from .logger import (
    AiLogFastStreamRedirect2File,
    StreamRedirector,
    logger
)
from .cloud_adapter import (
    Local2ObsMonitor,
    Obs2Local,
    cloud_monitor,
    mox_adapter
)
from .register import (
    ActionDict,
    DictConfig,
    MindFormerConfig,
    MindFormerModuleType,
    MindFormerRegister
)
from .utils import (
    DEBUG_INFO_PATH,
    MODE,
    PARALLEL_MODE,
    check_in_modelarts,
    check_shared_disk,
    count_params,
    get_output_root_path,
    get_output_subpath,
    set_output_path,
    set_strategy_save_path,
    str2bool
)
from .generic import add_model_info_to_auto_map
from .hub import (
    ENV_VARS_TRUE_VALUES,
    HubConstants,
    OPENMIND_CO_RESOLVE_ENDPOINT,
    OPENMIND_DYNAMIC_MODULE_NAME,
    PushToHubMixin,
    SESSION_ID,
    TIME_OUT_REMOTE_CODE,
    cached_file,
    check_imports,
    convert_file_size_to_int,
    create_dynamic_module,
    custom_object_save,
    download_url,
    extract_commit_hash,
    get_cached_module_file,
    get_checkpoint_shard_files,
    get_class_from_dynamic_module,
    get_class_in_module,
    get_file_from_repo,
    get_imports,
    get_relative_import_files,
    get_relative_imports,
    has_file,
    http_user_agent,
    init_mf_modules,
    is_offline_mode,
    is_remote_url,
    resolve_trust_remote_code
)

__all__ = []
__all__.extend(register.__all__)
