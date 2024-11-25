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
"""transform ckpt utils"""
import os
import shutil

from mindformers.tools.logger import logger


def show_progress(progress, prefix=''):
    """Show progress."""
    show_str = ('|%%-%ds|' % 50) % (int(50 * progress / 100) * "▮")
    logger.info("%s: %s%d%%", prefix, show_str, progress)


def make_soft_link(soft_link, ckpt_file, target_is_directory=False):
    """Make softlink to fit format of ms.load_checkpoint"""
    # Remove existing soft link if exists
    if os.path.islink(soft_link):
        os.unlink(soft_link)
        logger.info("Soft link '%s' has been removed.", soft_link)

    # Create directory for soft link if it doesn't exist
    os.makedirs(os.path.dirname(soft_link), exist_ok=True)

    # Create soft link
    try:
        os.symlink(ckpt_file, soft_link, target_is_directory=target_is_directory)
        logger.info("Soft link of checkpoint file from %s to %s created.", ckpt_file, soft_link)
    except FileExistsError:
        # If soft link creation fails due to FileExistsError, remove existing target and retry
        if target_is_directory:
            shutil.rmtree(soft_link)
        else:
            os.remove(soft_link)
        os.symlink(ckpt_file, soft_link, target_is_directory=target_is_directory)
        logger.info("Existing soft link '%s' removed and new soft link created.", soft_link)


def check_path(path, info="path"):
    """Check path exists."""
    if not isinstance(path, str):
        raise ValueError(f"`{info}` should be a string, but get {path}.")
    if not os.path.exists(path):
        raise ValueError(f"`{info}`={path} is not found!")


def check_rank_folders(path, rank_id):
    """Check if the folders in path are correct"""
    folder_name = "rank_{}".format(rank_id)
    if not os.path.exists(os.path.join(path, folder_name)):
        return False
    return True


def check_ckpt_file_exist(path):
    """Check if the files in path endswith .ckpt"""
    for file_name in os.listdir(path):
        if file_name.endswith('.ckpt'):
            return True
    return False


def is_power_of_two(number):
    """Checks if a number is a positive integer and a power of two."""
    return number > 0 and (number & (number - 1)) == 0
