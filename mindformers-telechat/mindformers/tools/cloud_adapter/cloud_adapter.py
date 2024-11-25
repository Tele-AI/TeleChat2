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
"""Cloud Adapter."""
import os
import time

from mindspore.train.callback import Callback

from ..logger import logger
from ..utils import check_obs_url, check_in_modelarts, \
    Validator, sync_trans, get_real_rank

if check_in_modelarts():
    import moxing as mox


__all__ = ['Local2ObsMonitor', 'Obs2Local', 'mox_adapter']


class Local2ObsMonitor(Callback):
    """File saved from local system to OBS server.

    Args:
        src_dir (str): Local system path, which means path of AI computing center platform.
        target_dir (str): OBS path starting with S3 or obs is used to save files.
        rank_id (int): the device's contents will be saved according to the actual rank_id.
            Default: None, means only the contents of the first device of each node are saved.
        upload_frequence (int): How often files are saved in AI computing center platform.
            Default: -1.
        keep_last (bool): Check whether files in the OBS are consistent with AI computing center platform.
            Default: True, means old file will be removed.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.
        log (logger): Use the log system to print information.
            Default: logging class for Python.
    """
    def __init__(self,
                 src_dir,
                 target_dir,
                 step_upload_frequence: int = 100,
                 epoch_upload_frequence: int = -1,
                 keep_last=True,
                 retry=3,
                 retry_time=5,
                 log=logger):
        super(Local2ObsMonitor, self).__init__()
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.step_upload_frequence = step_upload_frequence
        self.epoch_upload_frequence = epoch_upload_frequence
        self.keep_last = keep_last
        self.rank_id = get_real_rank()
        self.retry_time = retry_time
        self.retry = retry
        self.log = log
        self.cb_params = None
        self.pro = None
        self.on_modelarts = check_in_modelarts()
        if self.on_modelarts:
            check_obs_url(target_dir)

    def step_end(self, run_context):
        """upload files at the end of step."""
        if not self.on_modelarts:
            return
        if self.step_upload_frequence <= 0:
            return
        self.cb_params = run_context.original_args()
        if self.cb_params.cur_step_num % self.step_upload_frequence == 0 and os.listdir(self.src_dir):
            self.log.info("Starting upload output file to obs!")
            self.upload()

    def epoch_end(self, run_context):
        """upload files at the end of epoch."""
        if not self.on_modelarts:
            return
        if self.epoch_upload_frequence <= 0:
            return
        self.cb_params = run_context.original_args()
        if self.cb_params.cur_epoch_num % self.epoch_upload_frequence == 0 and os.listdir(self.src_dir):
            self.log.info("Starting upload output file to obs!")
            self.upload()

    def upload(self):
        """Upload Files to OBS."""
        if self.pro:
            # wait for last upload done
            self.pro.join()
        self.pro = self.sync2obs(self.src_dir, self.target_dir)

    @sync_trans
    def sync2obs(self, src_dir, target_dir):
        """Asynchronous transfer to OBS."""
        sub_dir_list = os.listdir(src_dir)
        for sub_dir in sub_dir_list:
            if 'checkpoint' in sub_dir:
                # for checkpoint, all processes upload its own checkpoint
                scr_ckpt_dir = os.path.join(src_dir, sub_dir, f"rank_{self.rank_id}")
                target_ckpt_dir = os.path.join(target_dir, sub_dir, f"rank_{self.rank_id}")
                if self.keep_last and mox.file.exists(target_ckpt_dir):
                    mox.file.remove(target_ckpt_dir, recursive=True)
                mox_adapter(scr_ckpt_dir, target_ckpt_dir, self.retry, self.retry_time, self.log)
            elif self.rank_id % 8 == 0:
                # for other outputs, one rank upload all
                mox_adapter(os.path.join(src_dir, sub_dir),
                            os.path.join(target_dir, sub_dir),
                            self.retry, self.retry_time, self.log)


class Obs2Local:
    """File saved from OBS server to local system of AI computing center platform.

    Args:
        rank_id (int): The obs's contents will be upload according to the actual rank_id value.
            Default: 0, means stored only one OBS file each node.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.
        log (logger): Use the log system to print information.
            Default: logging class for Python.
    """
    def __init__(self, rank_id=0, retry=3, retry_time=5, log=logger):
        self.rank_id = int(rank_id)
        self.retry_time = retry_time
        self.retry = retry
        self.log = log

    def obs2local(self, obs_url, local_url, special_id=None):
        """Pull Obs Files to Local."""
        obs_name = obs_url.split("/")[-1]
        mox_lock = os.path.join(local_url, "mox_copy_{}.lock".format(obs_name))
        local_url = os.path.join(local_url, obs_name)
        if special_id is None:
            if self.rank_id % 8 == 0:
                mox_adapter(obs_url, local_url, self.retry, self.retry_time, self.log)
                try:
                    os.mknod(mox_lock)
                except IOError:
                    pass
            else:
                self.log.info("programming sleep for waiting download file from obs to local.")
                while True:
                    if os.path.exists(mox_lock):
                        break
                    time.sleep(1)
        else:
            Validator.check_type(special_id, int)
            if self.rank_id == special_id:
                mox_adapter(obs_url, local_url, self.retry, self.retry_time, self.log)
                try:
                    os.mknod(mox_lock)
                except IOError:
                    pass
            else:
                self.log.info("programming sleep for waiting download file from obs to local.")
                while True:
                    if os.path.exists(mox_lock):
                        break
                    time.sleep(1)
        return local_url


def mox_adapter(src_dir, target_dir, retry=3, retry_time=5, log=logger):
    """File interaction with Moxing."""
    success = False
    for i in range(retry + 1):
        start = time.time()
        try:
            mox.file.copy_parallel(src_url=src_dir, dst_url=target_dir, threads=0, is_processing=False)
        except (FileNotFoundError, RuntimeError) as e:
            log.info("%s, from %s download to %s failed, will retry(%d) again.",
                     e, src_dir, target_dir, i)
            # sleep due to restriction of obs
            log.info("sleep time %d for waiting download file from obs.", retry_time)
            continue
        end = time.time()
        if Validator.is_obs_url(target_dir):
            if mox.file.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
        else:
            if os.path.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
    return success
