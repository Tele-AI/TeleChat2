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
"""Get resume ckpt."""
import os
import json

from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    create_file,
    check_in_modelarts,
    check_ckpt_file_name,
    get_real_rank,
    get_real_group_size,
    get_epoch_and_step_from_ckpt_name,
    get_rank_id_from_ckpt_name,
    get_remote_save_url,
    replace_rank_id_in_ckpt_name,
    remake_folder,
    is_publicly_accessible_path,
    get_output_root_path,
    barrier_world,
    Validator,
)

if check_in_modelarts():
    import moxing as mox

NO_META = "FOUND NO META.JSON"


def get_resume_checkpoint(checkpoint_dir, resume_training, resume_by_meta=True):
    """get resume checkpoint."""
    rank_id = get_real_rank()
    if isinstance(resume_training, str):
        if not resume_training.endswith(".ckpt"):
            resume_training = resume_training + ".ckpt"
        resume_training = replace_rank_id_in_ckpt_name(resume_training, rank_id)
        logger.info("Specify resume checkpoint: %s", \
            os.path.join(checkpoint_dir, f"rank_{rank_id}", resume_training))
        return resume_training

    if not is_publicly_accessible_path(checkpoint_dir):
        return True

    if not resume_by_meta:
        return True

    resume_ckpt = get_resume_checkpoint_by_meta(checkpoint_dir)
    return resume_ckpt


def get_resume_checkpoint_by_meta(checkpoint_dir):
    """get resume checkpoint by meta."""
    rank_id = get_real_rank()
    device_num = get_real_group_size()

    if check_in_modelarts():
        resume_record_dir = os.path.join(get_remote_save_url(), "resume_record")
        if not Validator.is_obs_url(resume_record_dir):
            raise ValueError(f"{resume_record_dir} is not a valid obs path.")
    else:
        resume_record_dir = os.path.join(get_output_root_path(), "resume_record")
    remake_folder(resume_record_dir, permissions=0o750)
    latest_checkpointed_iteration_txt = os.path.join(resume_record_dir,
                                                     "latest_checkpointed_iteration.txt")

    resume_ckpt = None
    if rank_id == 0:
        try:
            # 1. get basic resume ckpt file
            last_epoch, last_step, last_ckpt_file = get_minimum_epoch_step_and_ckpt(checkpoint_dir)
            if last_epoch is None or last_step is None or last_ckpt_file is None:
                logger.info("No meta.json available and will use the checkpoints "
                            "from the last timestamp for resume training.")
                check_last_timestamp_checkpoints(checkpoint_dir)
                create_file(latest_checkpointed_iteration_txt, NO_META)
                resume_ckpt = True
            else:
                ckpt_prefix = last_ckpt_file.split("-")[0]
                last_ckpt_file = ckpt_prefix + "-" + str(last_epoch) + "_" + str(last_step) + ".ckpt"
                logger.info("Basic resume checkpoint: %s", last_ckpt_file)

                # 2. get ckpt files suitable for resume training per rank
                resume_ckpt_list = get_resume_ckpt_list(checkpoint_dir, last_ckpt_file, rank_id, device_num)
                create_file(latest_checkpointed_iteration_txt, resume_ckpt_list)
                logger.info("Get resume checkpoint: %s", resume_ckpt_list[-1])
                resume_ckpt = os.path.basename(resume_ckpt_list[-1])

        # pylint: disable=W0703
        except BaseException as e:
            create_file(latest_checkpointed_iteration_txt, f"Error: {str(e)}")

    barrier_world("Get resume checkpoint by rank 0.")

    if not resume_ckpt:
        # get resume ckpt from txt recorded by rank_0
        resume_ckpt = get_resume_ckpt(latest_checkpointed_iteration_txt, rank_id)

    return resume_ckpt


def get_resume_ckpt(latest_checkpointed_iteration_txt, rank_id):
    """get resume ckpt"""
    logger.info(f"Get resume checkpoint from {latest_checkpointed_iteration_txt}")

    if not check_in_modelarts():
        if not os.path.exists(latest_checkpointed_iteration_txt):
            raise ValueError(f"Can not find {latest_checkpointed_iteration_txt}")
        with open(latest_checkpointed_iteration_txt, 'r') as f:
            resume_info = [line.strip() for line in f.readlines()]
    else:
        if not mox.file.exists(latest_checkpointed_iteration_txt):
            raise ValueError(f"OBS: Can not find {latest_checkpointed_iteration_txt}")
        with mox.file.File(latest_checkpointed_iteration_txt, 'r') as f:
            resume_info = [line.strip() for line in f.readlines()]

    if resume_info[0] == NO_META:
        logger.info("No meta.json available and will use the checkpoints "
                    "from the last timestamp for resume training.")
        return True

    if resume_info[0].startswith("Error"):
        raise ValueError(f"Get resume-able checkpoint failed, due to {resume_info[0]}.")

    resume_ckpt = replace_rank_id_in_ckpt_name(resume_info[-1], rank_id)
    logger.info("Get resume checkpoint: %s", resume_ckpt)
    return resume_ckpt


def get_minimum_epoch_step_and_ckpt(checkpoint_dir):
    """
    Parse all the meta.json files under checkpoint_dir and
    return the minimum epoch, step and ckpt file.
    """
    last_epoch = None
    last_step = None
    last_ckpt_file = None
    for rank_id_tmp in range(get_real_group_size()):
        meta_json = os.path.join(checkpoint_dir, f"rank_{rank_id_tmp}", "meta.json")
        if not os.path.exists(meta_json):
            logger.warning("%s is not found.", meta_json)
            continue
        with open(meta_json, "r") as json_file:
            try:
                meta_data = json.load(json_file)
                if not meta_data:
                    logger.warning(f"Get nothing from {json_file}.")
                    continue
            # pylint: disable=W0703
            except BaseException as e:
                logger.warning(f"load {json_file} failed due to: {str(e)}")
                continue
        epoch = meta_data.get("last_epoch", None)
        step = meta_data.get("last_step", None)
        ckpt_file = meta_data.get("last_ckpt_file", None)
        if not check_meta_info(epoch, step, ckpt_file, meta_json):
            continue
        if last_epoch is None or epoch < last_epoch or \
            (epoch == last_epoch and step < last_step):
            last_epoch = epoch
            last_step = step
            last_ckpt_file = ckpt_file

    return last_epoch, last_step, last_ckpt_file


def get_resume_ckpt_list(checkpoint_dir, last_ckpt_file, rank_id, device_num):
    """
    get ckpts suitable for resuming, where their rank numbers are intact,
    epoch and step are consistent, and the path exists.
    """
    # get all valid ckpts where the epoch and step values are not greater than those of last_ckpt_file.
    ckpt_prefix = last_ckpt_file.split("-")[0]
    last_epoch, last_step = get_epoch_and_step_from_ckpt_name(last_ckpt_file)
    original_rank = get_rank_id_from_ckpt_name(last_ckpt_file)
    valid_ckpts = {}
    for rank_id_tmp in range(device_num):
        ckpt_prefix_tmp = ckpt_prefix.replace(f"rank_{original_rank}", f"rank_{rank_id_tmp}")
        checkpoint_rank_dir = os.path.join(checkpoint_dir, f"rank_{rank_id_tmp}")
        if not os.path.exists(checkpoint_rank_dir):
            raise RuntimeError(f"{checkpoint_rank_dir} is not found!")
        for ckpt_file in os.listdir(checkpoint_rank_dir):
            if ckpt_file.startswith(ckpt_prefix_tmp) and ckpt_file.endswith(".ckpt"):
                epoch, step = get_epoch_and_step_from_ckpt_name(ckpt_file)
                if epoch < last_epoch or (epoch == last_epoch and step <= last_step):
                    key = str(epoch) + '_' + str(step)
                    valid_ckpts[key] = [ckpt_file] if not valid_ckpts.get(key) \
                        else valid_ckpts[key] + [ckpt_file]

    # get ckpts suitable for resuming, where their rank numbers are intact,
    # epoch and step are consistent, and the path exists.
    resume_ckpt_list = []
    for key in valid_ckpts:
        if check_checkpoints_by_rank(valid_ckpts[key], device_num):
            ckpt_file = replace_rank_id_in_ckpt_name(valid_ckpts[key][0], rank_id)
            resume_ckpt = os.path.join(checkpoint_dir, f"rank_{rank_id}", ckpt_file)
            if not os.path.exists(resume_ckpt):
                raise ValueError(f"{resume_ckpt} is not found!")
            resume_ckpt_list.append(resume_ckpt)
    if not resume_ckpt_list:
        raise RuntimeError("No checkpoint could be resumed.")
    resume_ckpt_list.sort(key=get_epoch_and_step_from_ckpt_name)
    logger.info("Find resume-able checkpoints as follow:")
    for ckpt in resume_ckpt_list:
        logger.info(ckpt)

    return resume_ckpt_list


def check_checkpoints_by_rank(checkpoints, rank_size):
    """Check rank number of ckpt in checkpoints are intact."""
    if not checkpoints:
        return False
    checkpoints.sort(key=get_rank_id_from_ckpt_name)
    rank_id_set = set(map(get_rank_id_from_ckpt_name, checkpoints))
    if len(rank_id_set) == rank_size and max(rank_id_set) == rank_size - 1:
        return True

    ori_checkpoint = checkpoints[0]
    ori_rank_id = get_rank_id_from_ckpt_name(ori_checkpoint)
    for i in range(rank_size):
        if i not in rank_id_set:
            checkpoint = ori_checkpoint.replace(f"rank_{ori_rank_id}", f"rank_{i}")
            logger.warning("%s is not found.", checkpoint)
    return False


def check_meta_info(epoch, step, ckpt_file, meta_json):
    """Check meta info."""
    if not isinstance(epoch, int) or not isinstance(step, int):
        logger.warning(f"The last_epoch and last_step load from {meta_json} should be int, "
                       f"but get last_epoch: {epoch} and last_step: {step}.")
        return False
    if not isinstance(ckpt_file, str):
        logger.warning(f"The last_ckpt_file load from {meta_json} should be str, "
                       f"but get last_ckpt_file: {ckpt_file}.")
        return False
    if not check_ckpt_file_name(ckpt_file):
        logger.warning(f"The last_ckpt_file load from {meta_json} should in the format of "
                       "{prefix}-{epoch}_{step}.ckpt, and '/' can't in prefix. "
                       f"But get last_ckpt_file: {ckpt_file}.")
        return False
    return True


def get_last_checkpoint(checkpoint_dir):
    """Get last timestamp checkpoint under checkpoint_dir."""
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])


def check_last_timestamp_checkpoints(checkpoint_dir):
    """
    Verify that the prefix, epoch and step of the checkpoints from the last timestamp
    are equal across all rank folders in the checkpoint_dir directory.
    """
    compared_checkpoint_name = None
    compared_original_checkpoint_name = None
    for rank_id_tmp in range(get_real_group_size()):
        checkpoint_rank_dir = os.path.join(checkpoint_dir, f"rank_{rank_id_tmp}")
        last_checkpoint = get_last_checkpoint(checkpoint_rank_dir)
        if not last_checkpoint:
            raise ValueError(f"Checkpoint not found under {checkpoint_rank_dir}.")
        if check_ckpt_file_name(last_checkpoint):
            compared_original_checkpoint_name = os.path.split(last_checkpoint)[1]
            compared_checkpoint_name = replace_rank_id_in_ckpt_name(last_checkpoint, 0)
            break

    if compared_checkpoint_name is None:
        # No checkpoint follows the {prefix}-{epoch}_{step}.ckpt naming convention.
        return

    find_diff_ckpt = False
    for rank_id_tmp in range(get_real_group_size()):
        checkpoint_rank_dir = os.path.join(checkpoint_dir, f"rank_{rank_id_tmp}")
        last_checkpoint = get_last_checkpoint(checkpoint_rank_dir)

        if not check_ckpt_file_name(last_checkpoint):
            logger.error("Find checkpoint not follow the {prefix}-{epoch}_{step}.ckpt"
                         f" naming convention: {last_checkpoint}.")
            find_diff_ckpt = True
            continue

        original_checkpoint_name = os.path.split(last_checkpoint)[1]
        current_checkpoint_name = replace_rank_id_in_ckpt_name(last_checkpoint, 0)
        if not compared_checkpoint_name:
            compared_checkpoint_name = current_checkpoint_name
            compared_original_checkpoint_name = original_checkpoint_name
        elif compared_checkpoint_name != current_checkpoint_name:
            raise ValueError("Check the prefix, epoch and step of the checkpoints "
                             "from the last timestamp failed. Find two different checkpoints: "
                             f"{compared_original_checkpoint_name} and {original_checkpoint_name}."
                             "If you ensure that the checkpoints for a certain epoch and step exist "
                             "and are not corrupted across all rank folders, you can set "
                             "the `resume_training` parameter to the filename of that checkpoint, "
                             "such as: llama_7b_rank_0-3_2.ckpt.")
    if find_diff_ckpt:
        raise ValueError("Some checkpoints follow the {prefix}-{epoch}_{step}.ckpt naming convention,"
                         " while others do not.")
