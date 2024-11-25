# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024, All rights reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Hub utilities: utilities related to download and cache models
"""
# import json
import os
import sys
from pathlib import Path
import re
import tempfile
from typing import Dict, Optional, Union
from urllib.parse import urlparse
from uuid import uuid4
import warnings

import requests

from mindformers.tools.logger import logger
from ..generic import working_or_temp_dir
from ... import __version__

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

SESSION_ID = uuid4().hex

OPENMIND_DYNAMIC_MODULE_NAME = "openmind_modules"
_is_offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES
_staging_mode = os.environ.get("OPENMIND_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.openmind.cn" if _staging_mode else "https://openmind.cn"  # confirm real default endpoint address
OPENMIND_CO_RESOLVE_ENDPOINT = os.environ.get("MDS_ENDPOINT", _default_endpoint)


class HubConstants:
    try:
        from openmind_hub import OM_HOME, OM_HUB_CACHE
    except ImportError:
        OM_HOME = ""
        OM_HUB_CACHE = ""
    OM_MODULES_CACHE = os.getenv("OM_MODULES_CACHE", os.path.join(OM_HOME, "modules"))
    OPENMIND_CACHE = os.getenv("OPENMIND_CACHE", OM_HUB_CACHE)


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def is_offline_mode():
    return _is_offline_mode


# pylint: disable=W0613
def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"openmind/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    import mindspore
    ua += f"; mindspore/{mindspore.__version__}"
    # CI will set this value to True
    if os.environ.get("openmind_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Examples:
        >>> convert_file_size_to_int("1MiB")
        1048576
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2 ** 30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2 ** 20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2 ** 10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10 ** 9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10 ** 6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10 ** 3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


# pylint: disable=C0103
def get_checkpoint_shard_files(
        pretrained_model_name_or_path,
        index_filename,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        user_agent=None,
        revision=None,
        subfolder="",
        _commit_hash=None,
        **deprecated_kwargs,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a
        model ID on the
        Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full
    path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json
    from tqdm import tqdm
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")
    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    from openmind_hub.utils import EntryNotFoundError, OmHubHTTPError
    from openmind_hub import try_to_load_from_cache

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        except EntryNotFoundError as e:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            ) from e
        except OmHubHTTPError as e:
            raise EnvironmentError(
                f"We couldn't connect to '{OPENMIND_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            ) from e

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


# pylint: disable=C0103
# pylint: disable=W0613
def cached_file(
        path_or_repo_id: Union[str, os.PathLike],
        filename: str,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        subfolder: str = "",
        repo_type: Optional[str] = None,
        user_agent: Optional[Union[str, Dict[str, str]]] = None,
        _raise_exceptions_for_missing_entries: bool = True,
        _raise_exceptions_for_connection_errors: bool = True,
        _commit_hash: Optional[str] = None,
        **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on openmind.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions
            if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a
            file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            since we use a git-based system for storing models and other artifacts on openmind.cn,
            so `revision` can be any identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on openmind.cn,
            you can specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```
    """
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://openmind.cn/{path_or_repo_id}/{revision}' for available files."
                )
            return None
        return resolved_file

    from openmind_hub.utils import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        LocalEntryNotFoundError,
        EntryNotFoundError,
        OmHubHTTPError,
        OMValidationError
    )

    from openmind_hub import _CACHED_NO_EXIST, om_hub_download, try_to_load_from_cache


    if cache_dir is None:
        cache_dir = HubConstants.OPENMIND_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        # param `repo_type` is not supported now, remove it temporary
        resolved_file = try_to_load_from_cache(
            path_or_repo_id,
            full_filename,
            cache_dir=cache_dir,
            revision=_commit_hash,
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            if not _raise_exceptions_for_missing_entries:
                return None
            raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)  # noqa

    try:
        # Load from URL or cache if already cached
        resolved_file = om_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if not subfolder else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to request access at "
            f"https://openmind.cn/{path_or_repo_id} and pass a token having permission to this repo "
            "by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        # replace xxx to openmind.cn
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'xxxxxxxxxxxxx'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://openmind.cn/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        # We try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
        )
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        if not _raise_exceptions_for_missing_entries or not _raise_exceptions_for_connection_errors:
            return None
        raise EnvironmentError(
            f"We couldn't connect to '{OPENMIND_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file "
            f"named {full_filename}.\nCheckout your internet connection."
        ) from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = "main"
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://openmind.cn/{path_or_repo_id}/{revision}' for available files."
        ) from e
    except OmHubHTTPError as e:
        # First we try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
        )
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        if not _raise_exceptions_for_connection_errors:
            return None
        raise EnvironmentError(
            f"There was a specific connection error when trying to load {path_or_repo_id}:\n{e}"
        ) from e
    except OMValidationError as e:
        raise EnvironmentError(
            f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a "
            f"local folder or the repo_id of a model on the Hub."
        ) from e

    return resolved_file


def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes.
    Its only use is for deprecated behavior allowing to download config/models with a single url instead
    of using the Hub.

    Args:
        url (`str`): The url of the file to download.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    from openmind_hub import http_get

    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated. You should host your file "
        f"on the Hub instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        http_get(url, tmp_file, proxies=proxies)
        tmp_file_path = tmp_file.name
    return tmp_file_path


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    from openmind_hub import REGEX_COMMIT_HASH
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def get_file_from_repo(
        path_or_repo: Union[str, os.PathLike],
        filename: str,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        subfolder: str = "",
        **deprecated_kwargs,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on openmind.cn.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions
            if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such
            a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on openmind, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on openmind, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from openmind.cn and cache.
    tokenizer_config = get_file_from_repo("bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("xlm-roberta-base", "tokenizer_config.json")
    ```
    """
    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


def has_file(
        path_or_repo: Union[str, os.PathLike],
        filename: str,
        revision: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        **deprecated_kwargs,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist
    for this repo, but will return False for regular connection errors.

    </Tip>
    """
    from openmind_hub.utils import (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        om_raise_for_status,
        GatedRepoError
    )
    from openmind_hub import om_hub_url
    from openmind_hub.utils import build_om_headers

    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    url = om_hub_url(path_or_repo, filename=filename, revision=revision)
    headers = build_om_headers(token=token, user_agent=http_user_agent())

    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        om_raise_for_status(r)
        return True
    except GatedRepoError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://openmind.cn/{path_or_repo} and pass a token having permission to this repo "
            "by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{path_or_repo} is not a local folder or a valid repository name on 'https://openmind.cn'."
        ) from e
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://openmind.cn/{path_or_repo}' for available revisions."
        ) from e
    except requests.HTTPError:
        # We return false for EntryNotFoundError (logical) as well as any connection error.
        return False


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
            self,
            repo_id: str,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
        from openmind_hub import create_repo
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
            self,
            working_dir: Union[str, os.PathLike],
            repo_id: str,
            files_timestamps: Dict[str, float],
            commit_message: Optional[str] = None,
            token: Optional[Union[bool, str]] = None,
            create_pr: bool = False,
            revision: str = None,
            commit_description: str = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        from openmind_hub import CommitOperationAdd, create_commit, create_branch

        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]

        # filter for actual files + folders at the root level
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # go over individual files of folder
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)

        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")

        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            # create_pr=create_pr,
            # revision=revision,
        )

    def push_to_hub(
            self,
            repo_id: str,
            use_temp_dir: Optional[bool] = None,
            commit_message: Optional[str] = None,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            max_shard_size: Optional[Union[int, str]] = "5GB",
            create_pr: bool = False,
            safe_serialization: bool = True,
            revision: str = None,
            commit_description: str = None,
            save_json: bool = False,
            **deprecated_kwargs,
    ) -> str:
        """
        Upload the {object_files} to the OpenMind Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                Will default to `True` if `repo_url` is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created

        Examples:

        ```python
        from openmind import {object_class}

        {object} = {object_class}.from_pretrained("bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("openmind/my-finetuned-bert")
        ```
        """
        working_dir = repo_id.split("/")[-1]

        repo_id = self._create_repo(
            repo_id,
            private=private,
            token=token,
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size,
                                 safe_serialization=safe_serialization, save_json=save_json)

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )
