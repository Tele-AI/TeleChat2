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
"""transform ckpt tool"""
import os
import time
import argparse
import tempfile
from glob import glob

from typing import Optional

import mindspore as ms

from mindformers.tools.utils import (
    get_real_rank,
    get_real_group_size,
    check_in_modelarts,
    get_output_root_path,
    get_remote_save_url,
    get_device_num_per_node,
    create_file,
    delete_file,
    remake_folder,
    is_main_rank,
    format_path,
    barrier_world
)
from mindformers.tools.logger import logger
from mindformers.tools.cloud_adapter import mox_adapter
from mindformers.tools.ckpt_transform.utils import (
    check_path,
    check_rank_folders,
    check_ckpt_file_exist,
    is_power_of_two,
    show_progress,
    make_soft_link
)

if check_in_modelarts():
    import moxing as mox

__all__ = ['TransformCkpt']


class TransformCkpt:
    """Transform src_checkpoint from src_strategy to dst_strategy."""
    def __init__(self,
                 auto_trans_ckpt: bool = False,
                 rank_id: Optional[int] = None,
                 world_size: Optional[int] = None,
                 transform_process_num: int = 1,
                 transform_by_rank: bool = False,
                 npu_num_per_node: int = None):
        """
        Initializes the object.

        Args:
            auto_trans_ckpt (bool, optional): Switch for automatic checkpoint conversion. Defaults to False.
            rank_id (int, optional): The rank ID of the current process. Defaults to None.
            world_size (int, optional): Total number of processes, typically equal to dp * mp * pp,
                representing the total number of slices for the target checkpoint. Defaults to None.
            transform_process_num (int): Number of processes used for checkpoint conversion.
                Defaults to 1.
                - If transform_process_num = 1, single-process conversion is used,
                where only rank_0 is responsible for checkpoint conversion, and other processes wait
                until rank_0 finishes conversion.
                - If transform_process_num > 1, multiprocess conversion is used.
                For example, in an 8-card task with transform_process_num=2,
                rank_0 is responsible for checkpoint conversion of slices rank_0/1/2/3, and rank_4 is
                responsible for checkpoint conversion of slices rank_4/5/6/7, while other processes wait
                until rank_0/4 finishes conversion.
                - Note:
                ① The conversion time decreases as transform_process_num increases,
                but this also increases the host memory consumption during conversion.
                When there is insufficient host memory, transform_process_num needs to be reduced.
                ② transform_process_num must be divisible by the number of NPU cards, and must not
                exceed the maximum number of NPU cards.
            transform_by_rank (bool): Whether the mindspore.transform_checkpoint_by_rank is used for
                checkpoint transform. It will automatically be set to True when transform_process_num > 1.
            npu_num_per_node (int, optional): Number of NPUs contained in each node. Required for
                configuration in the ModelArts platform.

        Raises:
            AssertionError: If npu_num_per_node is not a power of 2.
            AssertionError: If transform_process_num is less than 1.
            AssertionError: If transform_process_num is not divisible by world_size.
            Warning: If transform_process_num is greater than world_size.
            Warning: If transform_process_num is greater than 1 and less than node_num
            when training on AICC platform.
        """
        self.world_size = world_size if world_size else get_real_group_size()
        self.rank_id = rank_id if rank_id else get_real_rank()
        self.is_main_rank = is_main_rank()
        self.npu_num_per_node = npu_num_per_node or get_device_num_per_node()
        self.node_num = self.world_size // self.npu_num_per_node
        if not is_power_of_two(self.npu_num_per_node):
            raise ValueError(
                f"The `npu_num_per_node` must be a power of 2, but get {npu_num_per_node}")

        # Before obtaining transform_rank_id_list, check 1 ≤ transform_process_num ≤ world_size.
        if transform_process_num < 1:
            raise ValueError("transform_process_num should not smaller than 1,"
                             f"but got {transform_process_num}.")
        if transform_process_num > self.world_size:
            logger.warning("transform_process_num: %d should not bigger than world_size: %d. \
                transform_process_num is set to %d.",
                           transform_process_num, self.world_size, self.world_size)
            transform_process_num = self.world_size
        if self.world_size % transform_process_num != 0:
            raise ValueError(f"transform_process_num: {transform_process_num} "
                             f"should be divided by world_size: {self.world_size}.")
        if check_in_modelarts() and 1 < transform_process_num < self.node_num:
            logger.warning("transform_process_num: %d should not smaller than \
                node_num = world_size // npu_num_per_node = %d when training on AICC. \
                    transform_process_num is set to node num = %d",
                           transform_process_num, self.node_num, self.node_num)
            transform_process_num = self.world_size // npu_num_per_node
        if check_in_modelarts() and transform_process_num == 1:
            # The 0th NPU of each node is responsible for transform all checkpoints.
            # For example, if world_size=16 and npu_num_per_node=8,
            # then transform_rank_id_list=[0,8].
            self.transform_rank_id_list = [i for i in range(0, self.world_size, self.npu_num_per_node)]
        else:
            # Obtain transform_rank_id_list. For example, if world_size=8 and transform_process_num=2,
            # then transform_rank_id_list=[0,4], means that the 0th rank and the 4th rank
            # responsible for transform checkpoints.
            self.transform_rank_id_list = \
                [i for i in range(0, self.world_size, self.world_size // transform_process_num)]
        self.transform_process_num = len(self.transform_rank_id_list)

        if auto_trans_ckpt:
            # Check if pipeline parallel is being used and
            # get default path of transformed_checkpoint_dir and dst_strategy_dir.
            self.use_pipeline = ms.get_auto_parallel_context("pipeline_stages") > 1
            self.transformed_checkpoint_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
            if self.world_size > 1:
                self.dst_strategy_dir = os.path.join(get_output_root_path(), "strategy")
            if check_in_modelarts():
                self.transformed_checkpoint_dir_obs = os.path.join(get_remote_save_url(), "transformed_checkpoint")
                if self.world_size > 1:
                    self.dst_strategy_dir_obs = os.path.join(get_remote_save_url(), "strategy")
        self.auto_trans_ckpt = auto_trans_ckpt

        self.transform_by_rank = transform_by_rank
        if transform_process_num > 1:
            self.transform_by_rank = True
        elif self.world_size == 1:
            self.transform_by_rank = False

        self.cache_list = []
        logger.info(f"rank_id: {self.rank_id}")
        logger.info(f"world_size: {self.world_size}")
        logger.info(f"transform_process_num: {self.transform_process_num}")
        logger.info(f"transform_rank_id_list: {self.transform_rank_id_list}")

    def __call__(self,
                 src_checkpoint: str,
                 dst_checkpoint_dir: Optional[str] = None,
                 src_strategy: Optional[str] = None,
                 dst_strategy: Optional[str] = None,
                 prefix: str = "checkpoint_") -> str:
        """
        Transform checkpoints.

        Args:
            src_checkpoint (str): Absolute path of the source checkpoint or directory path.
                If it's a complete checkpoint, provide the absolute path.
                If it's a distributed checkpoint, provide the directory path.
                Distributed checkpoints should be stored in the format `model_dir/rank_x/xxx.ckpt`.
                Specify the directory path as `model_dir`.
                If there are multiple ckpts under rank_x folder,
                the last sorted ckpt file name will be used for transformation.
            src_strategy (str, optional): Path to the distributed strategy file
            corresponding to the source checkpoint.
                If it's a complete checkpoint, leave it as None.
                If it's a distributed checkpoint with pipeline parallelism,
                provide the merged strategy file path or the directory path of distributed strategy files.
                If it's a distributed checkpoint without pipeline parallelism,
                provide the path of any `ckpt_strategy_rank_x.ckpt`.
            dst_checkpoint_dir (str, optional): Directory path to save the target checkpoints.
            dst_strategy (str, optional): Path to the distributed strategy file corresponding to the target checkpoint.
                If it's a complete checkpoint, leave it as None.
                If it's a distributed checkpoint with pipeline parallelism,
                provide the merged strategy file path or the directory path of distributed strategy files.
                If it's a distributed checkpoint without pipeline parallelism,
                provide the path of any `ckpt_strategy_rank_x.ckpt`.
            prefix (str, optional): Prefix for the target checkpoint filenames. Default is "checkpoint_".

        Returns:
            str: Directory path where the transformed checkpoints are saved.
        """
        # Check src_checkpoint is str and path is existed.
        check_path(src_checkpoint, info="src_checkpoint")

        # convert path to realpath
        src_checkpoint = format_path(src_checkpoint)
        dst_checkpoint_dir = format_path(dst_checkpoint_dir)
        src_strategy = format_path(src_strategy)
        dst_strategy = format_path(dst_strategy)

        if src_checkpoint.endswith("/"):
            src_checkpoint = src_checkpoint[:-1]
        if self.auto_trans_ckpt:
            # dst_checkpoint_dir and dst_strategy is not required in auto_trans_ckpt mode.
            if dst_checkpoint_dir:
                logger.warning("`dst_checkpoint` is invalid when auto_trans_ckpt=True.")
            if dst_strategy:
                logger.warning("`dst_strategy` is invalid when auto_trans_ckpt=True.")
            dst_checkpoint_dir = self.transformed_checkpoint_dir
            if self.world_size > 1:
                # The dst_strategy must exist in auto_trans_ckpt mode.
                check_path(self.dst_strategy_dir, info="dst_strategy_dir")
                logger.info("The strategy files under %s will be used as the `dst_strategy`.", self.dst_strategy_dir)
        elif not dst_checkpoint_dir:
            # If dst_checkpoint_dir is not set, dst_checkpoint_dir is set to work_dir/transformed_checkpoint.
            dst_checkpoint_dir = os.path.join(os.getcwd(), "transformed_checkpoint")
            logger.warning("`dst_checkpoint_dir` is set to %s.", dst_checkpoint_dir)

        # Get src_strategy and dst_strategy.
        src_strategy = self.get_strategy(src_strategy)
        dst_strategy = self.get_strategy(dst_strategy)
        if self.auto_trans_ckpt:
            # Modify dst_strategy in auto_trans_ckpt mode.
            # dst_strategy is None when world_size = 1.
            # dst_strategy is dst_strategy_dir/rank_x/ckpt_strategy_rank_x.ckpt when world_size > 1.
            if self.world_size > 1:
                dst_strategy_list = glob(os.path.join(self.dst_strategy_dir, f"*_rank_{self.rank_id}.ckpt"))
                if not dst_strategy_list:
                    raise RuntimeError(f"The `dst_strategy`={self.dst_strategy_dir} \
                        does not contain strategy file of rank_{self.rank_id}.")
                if len(dst_strategy_list) > 1:
                    raise RuntimeError(f"There can only be one strategy file corresponding to rank_{self.rank_id}, \
                        but multiple strategy files corresponding to rank_{self.rank_id} were found \
                            in {self.dst_strategy_dir}.")
                dst_strategy = dst_strategy_list[0]
            else:
                dst_strategy = None

            if check_in_modelarts():
                if not mox.file.exists(self.transformed_checkpoint_dir_obs):
                    raise ValueError(f"transformed_checkpoint_dir_obs: "
                                     f"{self.transformed_checkpoint_dir_obs} is not found!")
                if self.world_size > 1 and not mox.file.exists(self.dst_strategy_dir_obs):
                    raise ValueError(f"dst_strategy_dir_obs: {self.dst_strategy_dir_obs} is not found!")


            # Get final dst_strategy in auto_trans_ckpt mode.
            dst_strategy = self.get_dst_strategy(dst_strategy)

        with tempfile.TemporaryDirectory() as soft_link_dir:
            # Build soft link for src_checkpoint.
            self.build_soft_link_of_checkpoint(src_checkpoint, soft_link_dir)
            for ckpt_name in os.listdir(soft_link_dir):
                src_ckpt_dir = os.path.join(soft_link_dir, ckpt_name)
                # Clear dst_ckpt_dir.
                dst_ckpt_dir = os.path.join(dst_checkpoint_dir, ckpt_name)
                remake_folder(dst_ckpt_dir, permissions=0o750)
                if check_in_modelarts():
                    dst_ckpt_dir_obs = os.path.join(self.transformed_checkpoint_dir_obs, ckpt_name)
                    remake_folder(dst_ckpt_dir_obs)
                barrier_world(f"Remake {dst_ckpt_dir} by main rank.")

                logger.info("The transformed checkpoint will be saved under %s.", dst_ckpt_dir)
                self.transform_ckpt(
                    src_checkpoint=src_ckpt_dir,
                    dst_checkpoint_dir=dst_ckpt_dir,
                    src_strategy=src_strategy,
                    dst_strategy=dst_strategy,
                    prefix=prefix
                )

        self.clear_cache()
        return dst_checkpoint_dir

    def transform_ckpt(self,
                       src_checkpoint,
                       dst_checkpoint_dir,
                       src_strategy=None,
                       dst_strategy=None,
                       prefix="checkpoint_"):
        """Transform ckpt using mindspore.transform_checkpoint"""
        self.check_src_checkpoint_and_strategy(src_checkpoint, src_strategy)
        if src_strategy is None and dst_strategy is None:
            raise ValueError("`src_strategy` and `dst_strategy` cannot both be None!")
        if check_in_modelarts():
            dst_checkpoint_dir_obs = os.path.join(self.transformed_checkpoint_dir_obs,
                                                  os.path.basename(dst_checkpoint_dir))

        if self.rank_id in self.transform_rank_id_list:
            try:
                if not self.transform_by_rank:
                    self.transform_checkpoints(src_checkpoint,
                                               dst_checkpoint_dir,
                                               prefix,
                                               src_strategy,
                                               dst_strategy)
                else:
                    self.transform_checkpoint_by_rank(src_checkpoint,
                                                      dst_checkpoint_dir,
                                                      prefix,
                                                      src_strategy,
                                                      dst_strategy)
                logger.info(".........Transform succeed!.........")
                logger.info("The transformed checkpoint was saved to %s", dst_checkpoint_dir)
                if check_in_modelarts():
                    transform_succeed_txt = os.path.join(dst_checkpoint_dir_obs,
                                                         f'transform_succeed_rank_{self.rank_id}.txt')
                else:
                    transform_succeed_txt = os.path.join(dst_checkpoint_dir,
                                                         f'transform_succeed_rank_{self.rank_id}.txt')
                create_file(transform_succeed_txt)
            # pylint: disable=W0703
            except BaseException as e:
                logger.error(f".........Transform failed due to: {str(e)}.........")
                if check_in_modelarts():
                    transform_failed_txt = os.path.join(dst_checkpoint_dir_obs,
                                                        f'transform_failed_rank_{self.rank_id}.txt')
                else:
                    transform_failed_txt = os.path.join(dst_checkpoint_dir,
                                                        f'transform_failed_rank_{self.rank_id}.txt')
                create_file(transform_failed_txt, info=str(e))

        # Wait transform finished.
        self.wait_transform(dst_checkpoint_dir)

        if check_in_modelarts():
            self.send_transformed_checkpoint_to_obs(dst_checkpoint_dir)

    def transform_checkpoints(self,
                              src_checkpoint,
                              dst_checkpoint,
                              prefix,
                              src_strategy,
                              dst_strategy):
        """transform checkpoints using mindspore.transform_checkpoints"""
        os.makedirs(dst_checkpoint, exist_ok=True)
        logger.info(".........Transforming ckpt.........")
        logger.info("src_checkpoint: %s", src_checkpoint)
        logger.info("src_strategy: %s", src_strategy)
        logger.info("dst_checkpoint: %s", dst_checkpoint)
        logger.info("dst_strategy: %s", dst_strategy)
        ms.transform_checkpoints(src_checkpoint,
                                 dst_checkpoint,
                                 prefix,
                                 src_strategy,
                                 dst_strategy)

    def transform_checkpoint_by_rank(self,
                                     src_checkpoint,
                                     dst_checkpoint,
                                     prefix,
                                     src_strategy,
                                     dst_strategy):
        """transform checkpoints using mindspore.transform_checkpoint_by_rank"""
        for current_transform_rank_id in \
            range(self.rank_id, self.rank_id + self.world_size // self.transform_process_num):
            logger.info(".........Transforming Ckpt For Rank: %d.........", current_transform_rank_id)
            src_rank_list = ms.rank_list_for_transform(current_transform_rank_id,
                                                       src_strategy,
                                                       dst_strategy)
            checkpoint_file_map = {}
            for src_rank_id in src_rank_list:
                checkpoint_rank_dir = os.path.join(src_checkpoint, f"rank_{src_rank_id}")
                checkpoint_file_list = glob(os.path.join(checkpoint_rank_dir, "*.ckpt"))
                if not checkpoint_file_list:
                    raise ValueError(f"The checkpoint of rank_{src_rank_id} is not found!")
                checkpoint_file_list = sorted(checkpoint_file_list, key=os.path.getmtime)
                checkpoint_file_map[src_rank_id] = checkpoint_file_list[-1]
            save_checkpoint_dir = os.path.join(dst_checkpoint, "rank_{}".format(current_transform_rank_id))
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            save_checkpoint_path = os.path.join(save_checkpoint_dir,
                                                "{}.ckpt".format(prefix + str(current_transform_rank_id)))
            logger.info("rank_list: %s", src_rank_list)
            logger.info("checkpoint_file_map: %s", checkpoint_file_map)
            logger.info("save_checkpoint_path: %s", save_checkpoint_path)
            logger.info("src_strategy: %s", src_strategy)
            logger.info("dst_strategy: %s", dst_strategy)
            ms.transform_checkpoint_by_rank(current_transform_rank_id,
                                            checkpoint_file_map,
                                            save_checkpoint_path,
                                            src_strategy,
                                            dst_strategy)

    def build_soft_link_of_checkpoint(self, checkpoint, soft_link_dir):
        """Build softlink of src checkpoint"""
        if os.path.isdir(checkpoint) and not check_rank_folders(checkpoint, 0) and \
            not check_ckpt_file_exist(checkpoint):
            raise ValueError(f"No rank_0 folder or ckpt files are found under {checkpoint}.")
        if os.path.isfile(checkpoint) and not checkpoint.endswith('.ckpt'):
            raise ValueError(f"The value of load_checkpoint must be a folder or a file with suffix '.ckpt', "
                             f"but got {checkpoint}")

        if os.path.isdir(checkpoint):
            if check_rank_folders(checkpoint, 0):
                # Has rank_0 dir under checkpoint.
                if check_ckpt_file_exist(checkpoint):
                    logger.warning(f"Find both ckpt files and rank folder under {checkpoint}, "
                                   "the rank folder will be used for checkpoint transform.")
                soft_link = os.path.join(soft_link_dir, os.path.basename(checkpoint))
                make_soft_link(soft_link, checkpoint)
            else:
                # Has ckpt file under checkpoint.
                for ckpt_file in glob(os.path.join(checkpoint, "*.ckpt")):
                    ckpt_name = os.path.basename(ckpt_file)
                    soft_link = os.path.join(soft_link_dir, ckpt_name.split(".")[0], "rank_0", ckpt_name)
                    make_soft_link(soft_link, ckpt_file)
        else:
            # checkpoint is a ckpt file.
            checkpoint = os.path.realpath(checkpoint)
            ckpt_name = os.path.basename(checkpoint)
            soft_link = os.path.join(soft_link_dir, ckpt_name.split(".")[0], "rank_0", ckpt_name)
            make_soft_link(soft_link, checkpoint)

    def clear_cache(self):
        """Clear cache file"""
        if self.is_main_rank:
            for cache_file in self.cache_list:
                delete_file(cache_file)

    def get_strategy(self, strategy_path, rank_id=None):
        """Merge strategy if strategy path is dir

        Args:
            strategy_path (str): The path of strategy.
            rank_id (int): The rank id of device.

        Returns:
            None or strategy path
        """
        if not strategy_path or strategy_path == "None":
            return None

        if not os.path.exists(strategy_path):
            raise ValueError(f'strategy_path: {strategy_path} not found!')

        if os.path.isfile(strategy_path):
            return strategy_path

        if os.path.isdir(strategy_path):
            if rank_id:
                merge_path = os.path.join(strategy_path, f'merged_ckpt_strategy_by_rank_{rank_id}.ckpt')
            else:
                merge_path = os.path.join(strategy_path, f'merged_ckpt_strategy.ckpt')

            merged_succeed_txt = os.path.join(strategy_path, "merge_succeed.txt")
            if self.is_main_rank:
                if os.path.exists(merge_path):
                    logger.info("The merged strategy: %s has existed. \
                                It will be deleted and re-merge a new strategy.", merge_path)
                    os.remove(merge_path)
                ms.merge_pipeline_strategys(strategy_path, merge_path)
                create_file(merged_succeed_txt)
                self.cache_list.append(merged_succeed_txt)
            else:
                while True:
                    if os.path.exists(merged_succeed_txt):
                        break

            return merge_path

        return None

    def get_dst_strategy(self, dst_strategy):
        """Get src and dst strategy."""
        if self.world_size == 1:
            return None

        if not (dst_strategy.endswith(f"_rank_{self.rank_id}.ckpt") and
                os.path.exists(dst_strategy)):
            raise ValueError(f"dst_strategy: {dst_strategy} is not found!")


        logger.info(".........Collecting strategy.........")
        if check_in_modelarts():
            self.send_strategy_to_obs(dst_strategy)

        if not self.use_pipeline:
            logger.info("pipeline_stage = 1, strategy using %s", dst_strategy)
            return dst_strategy
        self.wait_collect_all_strategy()

        logger.info(".........All strategy as follow.........")
        dst_strategy_path_list = glob(os.path.join(self.dst_strategy_dir, "*_rank_*.ckpt"))
        dst_strategy_path_list.sort()
        for dst_strategy_path in dst_strategy_path_list:
            logger.info("strategy: %s", dst_strategy_path)
        logger.info(".........Collecting %d strategy.........", len(dst_strategy_path_list))

        # merge strategy if pipeline_stage > 1
        if self.is_main_rank:
            logger.info(".........Merging strategy.........")
            merged_strategy_path = self.get_strategy(self.dst_strategy_dir)
            logger.info(".........Merging succeed.........")
            if self.rank_id == 0 and check_in_modelarts():
                self.send_strategy_to_obs(merged_strategy_path)
        else:
            logger.info(".........Waiting merge strategy.........")
            merged_strategy_path = os.path.join(self.dst_strategy_dir, "merged_ckpt_strategy.ckpt")
            while True:
                if os.path.exists(merged_strategy_path):
                    break
            logger.info(".........Merging succeed.........")
        dst_strategy = merged_strategy_path

        return dst_strategy

    def check_src_checkpoint_and_strategy(self, src_checkpoint, src_strategy):
        """check src checkpoint and strategy"""
        check_path(src_checkpoint, "src_checkpoint")
        if not os.path.isdir(src_checkpoint) or not glob(os.path.join(src_checkpoint, "rank_*")):
            raise ValueError("The load_checkpoint must be a dir and "
                             "ckpt should be stored in the format of load_checkpoint/rank_x/xxx.ckpt,"
                             f"but get {src_checkpoint}.")
        # Check rank_dirs is continuous.
        # For example, rank_0, rank_1, rank_4 is not continuous because it is missing rank_3
        src_checkpoint_rank_dir_list = glob(os.path.join(src_checkpoint, "rank_*"))
        src_checkpoint_rank_id_list = [int(rank_dir.split("_")[-1]) for rank_dir in src_checkpoint_rank_dir_list]
        src_checkpoint_rank_id_list.sort()
        src_checkpoint_rank_num = len(src_checkpoint_rank_id_list)
        for i in range(src_checkpoint_rank_num):
            if src_checkpoint_rank_id_list[i] != i:
                raise FileNotFoundError(f"The rank_{i} folder was not found under src_checkpoint folder.")

        # A full checkpoint do not require a strategy.
        if len(src_checkpoint_rank_id_list) == 1 and src_strategy:
            logger.warning("The src_checkpoint is complete, `src_strategy` is invalid and set to None.")
            src_strategy = None
        # Distributed checkpoints must be accompanied by strategy.
        if len(src_checkpoint_rank_id_list) > 1 and src_strategy is None:
            raise ValueError("`src_strategy` should not be None when `src_checkpoint` is sliced.")

    def send_strategy_to_obs(self, strategy):
        """Local rank send strategy file to obs."""
        obs_strategy_path = os.path.join(self.dst_strategy_dir_obs, os.path.basename(strategy))
        mox.file.copy(strategy, obs_strategy_path)
        logger.info("Save %s to %s.", strategy, obs_strategy_path)

    def send_transformed_checkpoint_to_obs(self, dst_checkpoint_dir):
        """Local rank send transformed checkpoint to obs."""
        dst_checkpoint_dir_obs = os.path.join(self.transformed_checkpoint_dir_obs, os.path.basename(dst_checkpoint_dir))
        dst_checkpoint_rankdir_obs = os.path.join(dst_checkpoint_dir_obs, f'rank_{self.rank_id}')
        dst_checkpoint_rankdir = os.path.join(dst_checkpoint_dir, f'rank_{self.rank_id}')
        mox_adapter(dst_checkpoint_rankdir, dst_checkpoint_rankdir_obs)
        logger.info("Save %s to %s.", dst_checkpoint_rankdir, dst_checkpoint_rankdir_obs)

    def wait_collect_all_strategy(self):
        """Wait all strategy collect over"""
        last_obs_strategy_num = -1
        last_strategy_num = -1
        start_time = time.time()
        while True:
            if check_in_modelarts():
                obs_strategy_path_list = mox.file.glob(
                    os.path.join(self.dst_strategy_dir_obs, "ckpt_strategy_rank_*.ckpt"))
                obs_strategy_num = len(obs_strategy_path_list)
                progress = (obs_strategy_num / self.world_size) * 100
                if obs_strategy_num != last_obs_strategy_num:
                    show_progress(progress, prefix="Collecting strategy")
                    last_obs_strategy_num = obs_strategy_num
                if obs_strategy_num < self.world_size:
                    time.sleep(5)
                    continue
                if self.is_main_rank:
                    for obs_strategy_path in obs_strategy_path_list:
                        dst_strategy_path = os.path.join(self.dst_strategy_dir, os.path.basename(obs_strategy_path))
                        mox.file.copy(obs_strategy_path, dst_strategy_path)
                        logger.info(f"Send {obs_strategy_path} to {dst_strategy_path}.")

            dst_strategy_path_list = glob(os.path.join(self.dst_strategy_dir, "ckpt_strategy_rank_*.ckpt"))
            dst_strategy_num = len(dst_strategy_path_list)
            progress = (dst_strategy_num / self.world_size) * 100
            if dst_strategy_num != last_strategy_num:
                show_progress(progress, prefix="Collecting strategy")
                last_strategy_num = dst_strategy_num
            if dst_strategy_num < self.world_size:
                if time.time() - start_time > 7200:
                    raise TimeoutError("Timeout while collecting all strategy!")
                time.sleep(5)
            else:
                break

    def wait_transform(self, ckpt_dir):
        """wait all node transform over"""
        last_count = -1
        while True:
            if check_in_modelarts():
                transformed_ckpt_dir_obs = os.path.join(self.transformed_checkpoint_dir_obs, os.path.basename(ckpt_dir))
                transform_failed_txts = mox.file.glob(os.path.join(transformed_ckpt_dir_obs,
                                                                   f'transform_failed_rank_*.txt'))
                transform_succeed_txts = mox.file.glob(os.path.join(transformed_ckpt_dir_obs,
                                                                    f'transform_succeed_rank_*.txt'))
            else:
                transform_failed_txts = glob(os.path.join(ckpt_dir, f'transform_failed_rank_*.txt'))
                transform_succeed_txts = glob(os.path.join(ckpt_dir, f'transform_succeed_rank_*.txt'))
            if transform_failed_txts:
                raise ValueError(f"Transform failed, find {transform_failed_txts}.")
            current_count = len(transform_succeed_txts)
            progress = (current_count / self.transform_process_num) * 100
            if current_count != last_count:
                show_progress(progress, prefix="Transforming checkpoint")
                last_count = current_count
            if current_count < self.transform_process_num:
                time.sleep(5)
            else:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_checkpoint',
                        default="",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_checkpoint_dir',
                        default="",
                        type=str,
                        help='path where to save dst ckpt')
    parser.add_argument('--src_strategy',
                        default=None,
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_strategy',
                        default=None,
                        help='path of dst ckpt strategy')
    parser.add_argument('--prefix',
                        default='checkpoint_',
                        type=str,
                        help='prefix of transformed checkpoint')
    parser.add_argument('--rank_id',
                        default=0,
                        type=int,
                        help='rank id')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='world size')
    parser.add_argument('--transform_process_num',
                        default=1,
                        type=int,
                        help='transform process num')
    parser.add_argument('--transform_by_rank',
                        default=False,
                        type=bool,
                        help='transform checkpoint using mindspore.transform_checkpoint_by_rank')
    args = parser.parse_args()

    transform_ckpt = TransformCkpt(
        rank_id=args.rank_id,
        world_size=args.world_size,
        transform_process_num=args.transform_process_num,
        transform_by_rank=args.transform_by_rank
    )

    transform_ckpt(
        src_checkpoint=args.src_checkpoint,
        dst_checkpoint_dir=args.dst_checkpoint_dir,
        src_strategy=args.src_strategy,
        dst_strategy=args.dst_strategy,
        prefix=args.prefix
    )

    print("......Transform finished!......")
