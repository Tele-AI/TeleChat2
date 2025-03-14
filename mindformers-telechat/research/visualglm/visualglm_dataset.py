# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Causal Image Modeling Dataset."""
import copy
import os
import re

import numpy as np
from PIL import Image
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset import vision
from mindspore.dataset.vision.utils import Inter

from mindformers.dataset.base_dataset import BaseDataset
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.version_control import get_dataset_map


def get_input_data_batch_slice_map(input_ids, eod_token_id, dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Args:
        input_ids: the input token ids
        eod_token_id: the id for <EOD>
        dis: the slice value for each rank
        rank_id: the current rank id
    Returns:
        batch_input_ids: the input token ids
        batch_position_ids: the position ids cosidering eod reset
        batch_attention_mask: the attention mask considering eod reset
    """
    rank = int(rank_id)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i, _ in enumerate(input_ids):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find the index of <EOS>
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_token_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering <EOS>
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class VisualGLMDataset(BaseDataset):
    """
    Causal Language Model pretrain dataset.
    output input_ids columns

    Args:
        dataset_config (dict): Config for dataset.

    Returns:
        A dataset for CausalLanguageModelDataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_generation']['gpt2']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = CausalLanguageModelDataset(config.train_dataset_task.dataset_config)
    """

    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create VisualGLM Model Dataset.")
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        if dataset_config.data_loader.type != "MindDataset" and \
                dataset_config.data_loader.type != "TFRecordDataset":
            dataset = cls._process_raw_text_data(dataset_config)
        else:
            dataset = cls._process_mindrecord_data(dataset_config)

        type_cast_op = C.TypeCast(mstype.int32)
        if dataset_config.eod_reset:
            if cls._is_semi_full_batch() or cls._is_data_parallel():
                rank_id = 0
                dis = dataset_config.batch_size
            else:
                # Each card slice a small batch from the full batch
                dis = dataset_config.batch_size // device_num
                if dataset_config.batch_size % device_num != 0:
                    raise ValueError(
                        f"batch size {dataset_config.batch_size} should be a multiple of device number {device_num}."
                        " You should change the args: per_batch_size.")

            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns)
            map_func = lambda input_ids: get_input_data_batch_slice_map(input_ids,
                                                                        eod_token_id=dataset_config.eod_token_id,
                                                                        rank_id=rank_id,
                                                                        dis=dis)
            dataset = get_dataset_map(dataset, map_func,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)

            for input_arg in dataset_config.output_columns:
                if "image" in input_arg:
                    continue
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
            dataset = dataset.project(columns=dataset_config.input_columns)
            for input_arg in dataset_config.input_columns:
                if "image" in input_arg:
                    continue
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)

        dataset = dataset.repeat(dataset_config.repeat)

        return dataset

    @classmethod
    def _prepare_for_model(cls, dataset, dataset_config):
        """ preprocess for model """
        from mindformers import Blip2ImageProcessor
        tokenizer_config = dataset_config.tokenizer
        tokenizer = build_tokenizer(tokenizer_config)
        image_processor = Blip2ImageProcessor(224, interpolation="bicubic")
        image_processor.resize.resize = vision.transforms.Resize((224, 224), Inter.BICUBIC)
        input_columns = dataset_config.input_columns
        max_source_length = dataset_config.max_source_length
        max_target_length = dataset_config.max_target_length
        max_seq_length = max_source_length + max_target_length

        def sft_visualglm_map_func(img, prompt, label):
            """Prepare input data for model fine-tuning or evaluation."""
            img = str(img)
            prompt = str(prompt)
            label = str(label)

            image = image_processor(Image.open(img).convert("RGB"))
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * 32
            input2 = tokenizer.encode("</img>问：" + prompt + "\n答：", add_special_tokens=False)
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=label, add_special_tokens=False)
            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[: max_source_length - 1]
            if len(b_ids) > max_target_length - 2:
                b_ids = b_ids[: max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
            input_id_len = len(input_ids)
            context_length = input_ids.index(tokenizer.bos_token_id)
            labels = [-100] * context_length + input_ids[context_length:]
            pad_len = max_seq_length - input_id_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            ignore_pad_token_for_loss = False
            if ignore_pad_token_for_loss:
                labels = [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in labels]
            image = image.asnumpy()
            image = image.squeeze(0)
            position_id = cls._create_position_ids(np.array(input_ids))
            attention_mask = cls._get_masks(np.array(input_ids))

            return tuple([image, input_ids, labels, position_id, attention_mask])

        dataset = dataset.map(sft_visualglm_map_func,
                              input_columns=["img", "prompt", "label"],
                              output_columns=input_columns)
        return dataset

    @classmethod
    def _get_masks(cls, input_ids, bos_token_id=130004):
        """generate mask from input id"""
        batch_size = 1
        seq_length = input_ids.shape[0]
        input_ids = [input_ids]
        context_lengths = [list(seq).index(bos_token_id) for seq in input_ids]
        attention_mask = np.tril(np.ones((batch_size, seq_length, seq_length)))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = np.expand_dims(attention_mask, axis=1)
        attention_mask = np.array(attention_mask < 0.5, np.bool_).squeeze(0)
        return attention_mask

    @classmethod
    def _get_position_ids(cls, input_ids, mask_positions, use_gmasks=None,
                          bos_token_id=130004, position_encoding_2d=True):
        """generate position ids from input id and mask positions"""

        seq_length = input_ids.shape[0]
        if use_gmasks is None:
            use_gmasks = [False]
        mask = bos_token_id * np.ones(shape=(seq_length), dtype=np.int32)
        mask = np.equal(input_ids, mask)
        # 要求input_ids中有且仅有一个bos_token_id
        context_lengths = np.argwhere(mask)[:, -1]
        if position_encoding_2d:
            position_ids = np.arange(seq_length, dtype=np.int64)
            for i, context_length in enumerate(context_lengths):
                position_ids[context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, dtype=np.int64),
                np.arange(seq_length - context_length, dtype=np.int64) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0).squeeze()
            position_ids = np.stack((position_ids, block_position_ids), axis=0)
        else:
            position_ids = np.arange(seq_length, dtype=np.int64)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    @classmethod
    def _create_position_ids(cls, input_ids, gmask_token_id=130001):
        """generate position ids from input id"""

        seq_length = input_ids.shape[0]
        seqs = input_ids
        # 要求input_ids中, 每行有且仅有一个gMASK
        use_gmasks = gmask_token_id * np.ones(shape=(seq_length), dtype=np.int32)
        mask = np.equal(seqs, use_gmasks)
        mask_positions = np.argwhere(mask)[:, -1]

        position_ids = cls._get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)
        return position_ids

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")

        tokenizer_config = dataset_config.tokenizer
        tokenizer = build_tokenizer(tokenizer_config)

        # 通过data_loader从数据集中加载数据
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'column_names': dataset_config.data_loader.column_names,
                                                      'tokenizer': tokenizer,
                                                      'scale': dataset_config.data_loader.scale,
                                                      'random_mapping': dataset_config.data_loader.random_mapping,
                                                      'shuffle': dataset_config.data_loader.shuffle})

        dataset = cls._prepare_for_model(dataset, dataset_config)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_files = []
        mind_compile = re.compile("mindrecord0*$")
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
