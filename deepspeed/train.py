# encoding=utf-8
# This code is based on the work of tatsu-lab/stanford_alpaca and has been modified from its original version.
# see: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import os
import copy
import json
import torch
import random
import shutil
import hashlib
import datasets
import transformers
import pandas as pd
import torch.nn as nn
import multiprocessing
import torch.nn.functional as F

from tqdm import tqdm
from string import Template
from itertools import chain
from transformers import Trainer
from torch.nn.functional import pad
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from accelerate.utils import DistributedType
from transformers.trainer_pt_utils import LabelSmoother
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# logger
_USING_LOGURU = False
try:
    from loguru import logger
    _USING_LOGURU = True
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s:  %(message)s"
    )
    logger = logging.getLogger(__name__)


# ================
#    constant
# ================
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_PAD_TOKEN = "<_pad>"
DEFAULT_BOS_TOKEN = "<_start>"
DEFAULT_EOS_TOKEN = "<_end>"
DEFAULT_UNK_TOKEN = "<_unk>"
DEFAULT_USR_TOKEN = "<_user>"
DEFAULT_BOT_TOKEN = "<_bot>"
DEFAULT_SYS_TOKEN = "<_system>"
DEFAULT_BOC_TOKEN = "<tool_call>"
DEFAULT_EOC_TOKEN = "</tool_call>"
DEFAULT_BOR_TOKEN = "<tool_response>"
DEFAULT_EOR_TOKEN = "</tool_response>"
LOCAL_RANK = None
NUM_CPU_CORES = min(multiprocessing.cpu_count(), 16) # 可根据训练环境适当更改并发数量

SYSTEM_TEMPLATE = "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。\n"

# ================
#    arguments
# ================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="TeleAI/TeleChat2-7B")


@dataclass
class DataArguments:
    data_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Configuration file for training data."
        }
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    max_seq_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }
    )
    train_mode: str = field(
        default="full",
        metadata={
            "help": "Train model with full parameters, lora or qlora.",
            "choices": [
                "full",
                "lora",
                "qlora"
            ]
        }
    )
    task_type: str = field(
        default="sft",
        metadata={
            "help": "Pre-train (continue-train) model or supervised-finetune model.",
            "choices": [
                "sft",
                "pretrain"
            ]
        }
    )
    group_pretrain_data: bool = field(
        default=False,
        metadata={
            "help": "Group pre-train data to max_seq_length if set to `True`."
        }
    )


@dataclass
class LoraArguments:
    lora_rank: int = field(
        default=64,
        metadata={
            "help": "Lora attention dimension (the \"rank\")."
        }
    )
    lora_alpha: int = field(
        default=16,
        metadata={
            "help": "The alpha parameter for Lora scaling."
        }
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={
            "help": "The dropout probability for Lora layers."
        }
    )


# ================
#    datasets
# ================
def sample_data_by_weights(dataset, weight):
    # 根据权重对数据进行采样
    input_ids = dataset["train"]["input_ids"]
    data_idx = [x for x in range(len(input_ids))]
    total_sample_data = int(len(data_idx) * weight)
    chosen_idx = []
    for idx in range(total_sample_data):
        cur_idx = idx % len(data_idx)
        if cur_idx == 0:
            random.shuffle(data_idx)
        chosen_idx.append(data_idx[cur_idx])
    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict(
            {
                "input_ids": [
                    input_ids[idx] for \
                        idx in chosen_idx
                ]
            }
        )
    })
    return dataset


def load_sft_dataset(data_args, training_args, tokenizer):
    """
    加载 sft 数据集
    """
    # special tokens
    sys_token_id = tokenizer(DEFAULT_SYS_TOKEN).input_ids
    eos_token_id = tokenizer(DEFAULT_EOS_TOKEN).input_ids
    usr_token_id = tokenizer(DEFAULT_USR_TOKEN).input_ids
    bot_token_id = tokenizer(DEFAULT_BOT_TOKEN).input_ids

    def _process_packing_dataset(all_messages, system=None):
        if system is None:
            system = SYSTEM_TEMPLATE
        else:
            system = system[0]
    
        outputs = []
        output = sys_token_id + tokenizer(system).input_ids + tokenizer("\n").input_ids
        length = len(output)
        for messages in all_messages:
            temp_output = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    temp_output += usr_token_id + tokenizer(content).input_ids
                elif role == "bot":
                    temp_output += bot_token_id + tokenizer(content).input_ids \
                        + eos_token_id + tokenizer("\n").input_ids
            temp_length = len(temp_output)

            if length + temp_length > training_args.max_seq_length:
                outputs.append(output[:-len(tokenizer("\n").input_ids)])
                # 重新积累数据
                output = sys_token_id + tokenizer(system).input_ids + tokenizer("\n").input_ids
                length = len(output)
            else:
                output += temp_output
                length += temp_length
        # 处理末尾数据
        if len(output) != 0:
            outputs.append(output)
        return {"input_ids": outputs}

    def _process_unpacking_dataset(all_messages, system=None):
        if system is None:
            system = [SYSTEM_TEMPLATE for _ in range(len(all_messages))]

        outputs = []
        for system_content, messages in zip(system, all_messages):
            input_ids = _transfer_chat_messages_into_tokens(messages, system_content)
            outputs.append(input_ids)
        return {"input_ids": outputs}

    def _transfer_chat_messages_into_tokens(messages, system):
        # 将 chat messages 处理成 token
        if hasattr(tokenizer, "apply_chat_template") \
            and tokenizer.chat_template is not None:
            messages = [{"role": "system", "content": system}] + messages
            prompt_token = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=False
            )
        else:
            # 拼装 prompt tokens
            prompt_token = sys_token_id + tokenizer(system).input_ids + tokenizer("\n").input_ids
            for m_idx, message in enumerate(messages, start=1):
                role = message["role"]
                content = message["content"]
                if role == "user":
                    prompt_token += usr_token_id + tokenizer(content).input_ids
                elif role == "bot":
                    if m_idx < len(messages):
                        prompt_token += bot_token_id + tokenizer(content).input_ids \
                            + eos_token_id + tokenizer("\n").input_ids
                    else:
                        prompt_token += bot_token_id + tokenizer(content).input_ids \
                            + eos_token_id
        return prompt_token

    # 加载数据配置文件
    data_config = read_json(data_args.data_config_file)
    # 创建数据缓存路径
    cache_dir = os.path.join(training_args.output_dir, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 扫描数据，保留可以正确加载的数据
    with training_args.main_process_first("processing data"):
        loaded_dataset = [] # 所有数据
        # 加载数据
        loading_pipeline = tqdm(data_config.items())
        for idx, (data_path, data_weight) in enumerate(loading_pipeline):
            if not data_path.endswith("jsonl"):
                # 非jsonl文件，跳过处理
                continue

            file_name = os.path.basename(data_path).replace(".jsonl", "")
            loading_pipeline.set_description(f"Loading: {file_name}.jsonl")
            # 加载单个jsonl文件，并按照data_weight进行数据采样
            dataset = datasets.load_dataset(
                "json",
                data_files=data_path,
                cache_dir=os.path.join(cache_dir, "raw_data", file_name),
                keep_in_memory=False
            )
            dataset = dataset["train"].to_pandas()
            dataset = dataset.sample(frac=data_weight, replace=True)

            if idx == 0:
                loaded_dataset = dataset
            else:
                assert (loaded_dataset.columns == dataset.columns).all()
                loaded_dataset = pd.concat([loaded_dataset, dataset])

        # 微调数据拼接
        # 非工具调用且非多论文对话数据进行拼接
        rank0_logging("Start processing data!")
        loaded_dataset = loaded_dataset.reset_index(drop=True)
        packing_dataset = loaded_dataset[loaded_dataset["tool"] == False]
        packing_dataset = packing_dataset[packing_dataset["multiturn"] == False]
        unpacking_dataset = loaded_dataset[~loaded_dataset.index.isin(packing_dataset.index)]
        
        # 处理需要拼接的数据
        train_dataset = []
        for idx, (system, grouped_dataset) in enumerate(packing_dataset.groupby("system")):
            if len(grouped_dataset) > 0:
                grouped_dataset = datasets.Dataset.from_dict({
                    "messages": grouped_dataset["dialog"].values,
                    "system": [system for _ in range(len(grouped_dataset))]
                })
                packing_cache_path = os.path.join(cache_dir, "packing_data", get_md5(system))
                os.makedirs(packing_cache_path, exist_ok=True)
                grouped_dataset = grouped_dataset.map(
                    function=_process_packing_dataset,
                    input_columns=["messages", "system"],
                    remove_columns=["messages", "system"],
                    batched=True,
                    num_proc=NUM_CPU_CORES,
                    load_from_cache_file=True,
                    cache_file_name=os.path.join(packing_cache_path, "tokenized.arrow"),
                    keep_in_memory=False
                )
                # 拼接数据
                if idx == 0:
                    train_dataset = grouped_dataset
                else:
                    assert train_dataset.features.type == grouped_dataset.features.type
                    train_dataset = datasets.concatenate_datasets([
                        train_dataset,
                        grouped_dataset
                    ])

        # 处理不需要拼接的数据
        if len(unpacking_dataset) > 0:
            unpacking_dataset = datasets.Dataset.from_dict({
                    "messages": unpacking_dataset["dialog"].values,
                    "system": unpacking_dataset["system"].values
                })
            unpacking_cache_path = os.path.join(cache_dir, "unpacking_data")
            os.makedirs(unpacking_cache_path, exist_ok=True)
            unpacking_dataset = unpacking_dataset.map(
                function=_process_unpacking_dataset,
                input_columns=["messages", "system"],
                remove_columns=["messages", "system"],
                batched=True,
                num_proc=NUM_CPU_CORES,
                load_from_cache_file=True,
                cache_file_name=os.path.join(unpacking_cache_path, "tokenized.arrow"),
                keep_in_memory=False
            )
            if type(train_dataset) == list:
                train_dataset = unpacking_dataset
            else:
                assert train_dataset.features.type == unpacking_dataset.features.type
                train_dataset = datasets.concatenate_datasets([
                    train_dataset,
                    unpacking_dataset
                ])

    return train_dataset


def load_pretrain_dataset(data_args, training_args, tokenizer):
    """
    加载 pretrain / continue-train 数据集
    """

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= training_args.max_seq_length:
            total_length = (total_length // training_args.max_seq_length) * training_args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + training_args.max_seq_length] for i in range(0, total_length, training_args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def _transfer_pretrain_prompt_into_tokens(samples):
        # 将 text 处理成 token
        outputs = []
        eos_token_id = tokenizer(DEFAULT_EOS_TOKEN).input_ids
        for sample in samples:
            prompt_token = tokenizer(sample).input_ids + eos_token_id
            outputs.append(prompt_token)
        # print(prompt_token)
        return {"input_ids": outputs}

    data_config = read_json(data_args.data_config_file)
    # 创建数据缓存路径
    cache_dir = os.path.join(training_args.output_dir, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 扫描数据，保留可以正确加载的数据
    with training_args.main_process_first("processing data"):
        train_dataset = [] # 所有数据
        data_process_pipeline = tqdm(data_config.items())
        for idx, (data_path, data_weight) in enumerate(data_process_pipeline):
            if not data_path.endswith("jsonl"):
                # 非jsonl文件，跳过处理
                continue

            file_name = os.path.basename(data_path).replace(".jsonl", "")
            data_process_pipeline.set_description(f"Processing: {file_name}.jsonl")
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            # 加载数据
            try:
                dataset = datasets.load_from_disk(
                    cache_path, keep_in_memory=False
                )
            except Exception:
                # 数据处理缓存路径
                temp_cache_path = os.path.join(cache_path, "temp")
                # 加载本地数据集
                prompt_dataset = datasets.load_dataset(
                    "json",
                    data_files=data_path,
                    cache_dir=temp_cache_path,
                    keep_in_memory=False
                )

                # 把文本处理成 token
                # 预训练、继续训练数据 处理成 token
                # 启用 batched=True 来处理数据
                # 数据将被分成 NUM_CPU_CORES 批进行处理
                # 随后可在每一个线程中将数据拼接到 max_seq_length
                # 但仍然可能出现拼接不足 max_seq_length 的情况
                dataset = prompt_dataset.map(
                    function=_transfer_pretrain_prompt_into_tokens,
                    batched=True,
                    input_columns="text",
                    remove_columns="text",
                    num_proc=NUM_CPU_CORES,
                    load_from_cache_file=True,
                    cache_file_names={
                        k: os.path.join(temp_cache_path, "tokenized.arrow") \
                            for k in prompt_dataset
                    },
                    keep_in_memory=False
                )
                # 将数据 group 到 max_seq_length
                if training_args.group_pretrain_data:
                    dataset = dataset.map(
                        function=group_texts,
                        batched=True,
                        num_proc=NUM_CPU_CORES,
                        load_from_cache_file=True,
                        cache_file_names={
                            k: os.path.join(temp_cache_path, 'grouped.arrow') \
                                for k in dataset},
                        keep_in_memory=False
                    )

                # 按照权重采样数据
                dataset = sample_data_by_weights(dataset, data_weight)
                # 缓存训练数据
                dataset.save_to_disk(cache_path)

            # 拼接数据
            if idx == 0:
                train_dataset = dataset["train"]
            else:
                assert train_dataset.features.type == dataset["train"].features.type
                train_dataset = datasets.concatenate_datasets([
                    train_dataset,
                    dataset["train"]
                ])
    
    return train_dataset


def load_train_dataset(data_args, training_args, tokenizer):
    """
    多线程处理数据为 token，并缓存下来
    """
    if training_args.task_type == "sft":
        train_dataset = load_sft_dataset(data_args, training_args, tokenizer)
    else:
        train_dataset = load_pretrain_dataset(data_args, training_args, tokenizer)

    rank0_logging(f"All data loaded! Total training number: {len(train_dataset)}")
    return train_dataset


class TelechatSFTDataCollator(object):
    # 在 Trainer 中处理每一个 batch 的数据
    # batch 数据来自于传入的 Dataset
    # 将 batch data 按照 user content 掩码的方式处理
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.eos_token_id = tokenizer(DEFAULT_EOS_TOKEN).input_ids[0]
        self.usr_token_id = tokenizer(DEFAULT_USR_TOKEN).input_ids[0]
        self.bot_token_id = tokenizer(DEFAULT_BOT_TOKEN).input_ids[0]
        self.pad_token_id = tokenizer(DEFAULT_PAD_TOKEN).input_ids[0]

    def __call__(
        self,
        batch: List
    ) -> Dict:
        """处理一个batch的数据"""
        # 按照每个batch数据的最大长度或指定训练最大长度来截断batch数据
        # 避免训练时每次都是用max_seq_length降低训练性能
        batch_max_len = min(
            max([len(x.get("input_ids", [])) for x in batch]),
            self.max_seq_length
        )
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        batch_loss_mask = []
        # 处理数据
        for sample in batch:
            input_ids = sample.get("input_ids", [])
            attention_mask = [1] * len(input_ids)
            labels = copy.deepcopy(input_ids)

            # 避免空数据、异常数据
            if len(input_ids) == 0:
                continue

            # list 数据转成 tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
    
            # 按照 @世萱 老师的代码，将 user context 部分进行掩码
            # 找到需要计算loss的部分
            loss_mask = torch.zeros(labels.size())
            user_indices = torch.where(input_ids == self.usr_token_id)[0].tolist()
            bot_indices = torch.where(input_ids == self.bot_token_id)[0].tolist()
            eos_indices = torch.where(input_ids == self.eos_token_id)[0].tolist()

            # 当数据按照 max_seq_length 截断时，可能出现 <_end> token 被截断的情况
            # 此时 eos_indices 缺少一位，补足数据长度即可
            if len(eos_indices) + 1 == len(bot_indices):
                eos_indices.append(input_ids.size(0))
            
            for idx in range(len(bot_indices)):
                bot_idx = bot_indices[idx]
                usr_idx = user_indices[idx]
                eos_idx = eos_indices[idx]
                
                loss_mask[bot_idx:eos_idx + 1] = 1
                loss_mask[usr_idx] = 1

            # # -100 在loss计算时会被忽略
            # labels = torch.where(
            #     loss_mask == 1,
            #     input_ids,
            #     IGNORE_TOKEN_ID
            # )

            # padding & truncate
            length_to_padding = batch_max_len - len(input_ids)
            input_ids = pad(
                input_ids, [0, length_to_padding], "constant", self.pad_token_id
            )[:self.max_seq_length]
            labels = pad(
                labels, [0, length_to_padding], "constant", self.pad_token_id
            )[:self.max_seq_length]
            attention_mask = pad(
                attention_mask, [0, length_to_padding], "constant", 0
            )[:self.max_seq_length]
            loss_mask = pad(
                loss_mask, [0, length_to_padding], "constant", 0
            )[:self.max_seq_length]

            # reshape，便于concat
            batch_input_ids.append(input_ids.view(1, -1))
            batch_attention_mask.append(attention_mask.view(1, -1))
            batch_labels.append(labels.view(1, -1))
            batch_loss_mask.append(loss_mask.view(1, -1))

        return {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
            "loss_mask": torch.cat(batch_loss_mask, dim=0)
        }


class TelechatPretrainDataCollator(object):
    # 在 Trainer 中处理每一个 batch 的数据
    # batch 数据来自于传入的 Dataset
    # 将 batch data 按照 user content 掩码的方式处理
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.pad_token_id = tokenizer(DEFAULT_PAD_TOKEN).input_ids[0]

    def __call__(
        self,
        batch: List
    ) -> Dict:
        """处理一个batch的数据"""
        # 按照每个batch数据的最大长度或指定训练最大长度来截断batch数据
        # 避免训练时每次都是用max_seq_length降低训练性能
        batch_max_len = min(
            max([len(x.get("input_ids", [])) for x in batch]),
            self.max_seq_length
        )
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        # 处理数据
        for sample in batch:
            input_ids = sample.get("input_ids", [])
            attention_mask = [1] * len(input_ids)
            labels = copy.deepcopy(input_ids)

            # 避免空数据、异常数据
            if len(input_ids) == 0:
                continue

            # list 数据转成 tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)

            # padding & truncate
            length_to_padding = batch_max_len - len(input_ids)
            input_ids = pad(
                input_ids, [0, length_to_padding], "constant", self.pad_token_id
            )[:self.max_seq_length]
            labels = pad(
                labels, [0, length_to_padding], "constant", self.pad_token_id
            )[:self.max_seq_length]
            attention_mask = pad(
                attention_mask, [0, length_to_padding], "constant", 0
            )[:self.max_seq_length]

            # reshape，便于concat
            batch_input_ids.append(input_ids.view(1, -1))
            batch_attention_mask.append(attention_mask.view(1, -1))
            batch_labels.append(labels.view(1, -1))

        return {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "labels": torch.cat(batch_labels, dim=0)
        }


# ================
#     trainer
# ================
class MyTrainer(Trainer):
    # 重写 loss 计算函数
    # 采取 mask loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = None
        if "loss_mask" in inputs:
            loss_mask = inputs.pop("loss_mask")
            if "labels" in inputs:
                labels = inputs.pop("labels")
        else:
            loss_mask = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if loss_mask is not None:
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            # compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_loss_mask = loss_mask[..., 1:].contiguous()
            shift_logits = F.log_softmax(shift_logits, dim=-1)
            loss = -torch.gather(shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            shift_loss_mask = shift_loss_mask.view(-1)
            expected_number_of_tokens = shift_loss_mask.sum()
            loss = torch.sum(loss.view(-1) * shift_loss_mask) / expected_number_of_tokens
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


# ================
#    functions
# ================
def rank0_logging(*args):
    if LOCAL_RANK == 0:
        logger.info(*args)


def rank0_warning(*args):
    if LOCAL_RANK == 0:
        logger.warning(*args)


def get_md5(context):
    return hashlib.md5(context.encode(encoding='UTF-8')).hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf_8_sig") as file:
        return json.load(file)


def read_jsonl(path):
    with open(path, "r", encoding="utf_8_sig") as files:
        return [json.loads(line) for line in files]


def load_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    return tokenizer


def find_all_linear_modules(model):
    # This function is modified from https://github.com/yangjianxin1/Firefly 
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def load_model(model_args, training_args, lora_args):
    # This function is modified from https://github.com/QwenLM/Qwen
    torch_dtype = torch.float32
    if training_args.fp16:
        torch_dtype = torch.float16
    if training_args.bf16:
        torch_dtype = torch.bfloat16

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if training_args.train_mode == "qlora":
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )
    
    if (
        training_args.train_mode == "lora"
        and is_deepspeed_zero3_enabled()
        and training_args.task_type == "pretrain"
    ):
        raise RuntimeError(
            "ZeRO3 is incompatible with LoRA when finetuning on base model."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        low_cpu_mem_usage=not is_deepspeed_zero3_enabled()
    )

    if training_args.train_mode in ["lora", "qlora"]:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        target_modules = find_all_linear_modules(model)
        lora_config = LoraConfig(
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        if training_args.train_mode == "qlora":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        # 准备 lora model
        model = get_peft_model(model, lora_config)
        # Lora 模型参数信息
        model.print_trainable_parameters()
    
    total = sum(p.numel() for p in model.parameters())
    rank0_logging("Model loaded! Total model params: %.2fB" % (total / 1e9))
    return model


def save_trained_model(model_args, training_args, trainer):
    trainer.save_model(training_args.output_dir)
    # copy .py files
    if os.path.exists(training_args.output_dir):
        if os.path.exists(model_args.model_name_or_path):
            remote_codes = [
                file for file in os.listdir(model_args.model_name_or_path) \
                    if file.endswith(".py")
            ]
            for filename in remote_codes:
                if not os.path.exists(
                    os.path.join(training_args.output_dir, filename)
                ):
                    shutil.copy(
                        os.path.join(model_args.model_name_or_path, filename),
                        os.path.join(training_args.output_dir, filename)
                    )


def train():
    global LOCAL_RANK

    # 加载各类配置文件
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments
        )
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    LOCAL_RANK = training_args.local_rank

    # 新建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    # 设置log输出位置
    if _USING_LOGURU:
        logger.add(os.path.join(training_args.output_dir, "train.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"))
        logger.addHandler(file_handler)

    rank0_logging(f"Model arguments:\n{model_args}")
    rank0_logging(f"Data arguments:\n{data_args}")
    rank0_logging(f"Training arguments:\n{training_args}")
    if training_args.train_mode in ["lora", "qlora"]:
        rank0_logging(f"Lora arguments:\n{lora_args}")

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载 Tokenizer
    tokenizer = load_tokenizer(model_args)
    
    # 加载训练数据
    train_dataset = load_train_dataset(data_args, training_args, tokenizer)
    if training_args.task_type == "sft":
        data_collator = TelechatSFTDataCollator(
            tokenizer=tokenizer,
            max_seq_length=training_args.max_seq_length
        )
    else:
        data_collator = TelechatPretrainDataCollator(
            tokenizer=tokenizer,
            max_seq_length=training_args.max_seq_length
        )

    # 加载模型
    model = load_model(model_args, training_args, lora_args)

    # 设定 Trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 开始训练
    train_results = trainer.train()

    # 训练结束，保存 metrics & ckpt
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_state()
    save_trained_model(model_args, training_args, trainer)

# ================
#      main
# ================
if __name__ == "__main__":
    train()