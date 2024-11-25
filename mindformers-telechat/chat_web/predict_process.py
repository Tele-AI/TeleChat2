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
"""Predict process"""
import os
from typing import List
from multiprocessing import Queue, Process

import mindspore as ms
from mindspore.common import initializer as init

from config.server_config import default_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers import MindFormerConfig, build_context, BaseStreamer, build_parallel_config, logger, \
    AutoModel, AutoTokenizer, TextIteratorStreamer


def get_model(config: MindFormerConfig):
    """
    Get model

    Notice:
        You can modify this function to customize the server to run other model
        except those in the "Mindformers.models". (e.g. models in /research)
    """
    return AutoModel.from_config(config)


def get_tokenizer(config: MindFormerConfig):
    """
    Get tokenizer

    Notice:
        You can modify this function to customize the server to run other tokenizer
        except those in the "Mindformers.models". (e.g. models in /research)
    """
    return AutoTokenizer.from_pretrained(config.trainer.model_name)


def build_prompt(inputs: str):
    """Build prompt"""
    prompt = "{}"  # You can modify this to build prompt for your model input
    return prompt.format(inputs)


def build_multi_round(inputs, history):
    """Build multi round"""
    multi_round_prompt = ""  # You can modify this to build multi-round input for your model input
    prev_rounds = ""
    for i, (query, response) in enumerate(history):
        prev_rounds += multi_round_prompt.format(i, query, response)
    return prev_rounds + inputs


def generate_process(device_id: int,
                     device_num: int,
                     config: MindFormerConfig,
                     input_q: Queue,
                     output_q: Queue,
                     device_range: List = None,
                     streamer: BaseStreamer = None):
    """Generate process"""
    ms.set_context(device_id=device_id, device_target="Ascend", mode=ms.GRAPH_MODE)

    if device_num > 1:
        # set distributed env variable and build context
        os.environ['DEVICE_ID'] = str(device_id)
        os.environ['RANK_ID'] = str(device_id - device_range[0])
        logger.info(f"{device_id} card exported DEVICE_ID={str(device_id)}")
        logger.info(f"{device_id} card exported RANK_ID={str(device_id - device_range[0])}")
        build_context(config)
        build_parallel_config(config)
        logger.info(f"{device_id} card context config is: {config.parallel_config}")
        config.model.model_config.parallel_config = config.parallel_config

    # initialize model and tokenizer
    network = get_model(config)
    tokenizer = get_tokenizer(config)
    logger.info(f"{device_id} card model and tokenizer initialize success")

    if device_num > 1:
        # load distributed checkpoint
        model = ms.Model(network)
        seq_length = config.model.model_config.seq_length
        infer_data = ms.Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)
        logger.info(f"{device_id} card load checkpoint success")

    history = []
    while True:
        infer_data: dict = input_q.get()
        inputs = infer_data['inputs']
        infer_data.pop('inputs')
        is_stream = infer_data.pop('stream', None)

        prompted_inputs = build_prompt(inputs)
        inputs = build_multi_round(prompted_inputs, history)
        input_ids = tokenizer(inputs)["input_ids"]

        generation_kwargs = dict(
            input_ids=[input_ids]
        )
        generation_kwargs.update(**infer_data)
        if streamer is not None:
            generation_kwargs.update(streamer=streamer)

        output_ids = network.generate(**generation_kwargs)
        output_ids[-1] = output_ids[-1][len(input_ids):-1]
        output = tokenizer.decode(output_ids)[-1]

        logger.info(f"{device_id} card generate output: {output}")
        history.append((prompted_inputs, output))
        if not is_stream:
            output_q.put_nowait(output)
            logger.debug(f"{device_id} card put output into output queue.")


def update_checkpoint_config(config):
    """Update checkpoint config"""
    if not config.load_checkpoint:
        config.load_checkpoint = ""
    if config.auto_trans_ckpt:
        raise ValueError("auto_trans_ckpt does not support in chat web server. "
                         "Please using transformed ckpt and set auto_trans_ckpt to False.")
    if os.path.isdir(config.load_checkpoint) and config.use_parallel:
        config.model.model_config.checkpoint_name_or_path = None


class MindFormersInfer:
    """MindFormers Infer"""

    def __init__(self) -> None:
        self.input_q_list = []
        self.output_q_list = []

        model_config = default_config['model']['config']
        self.device_num = default_config['model']['device_num']
        self.device_id = default_config['model']['device_id']
        self.device_range = None
        if self.device_num > 1:
            self.device_range = default_config['model']['device_range']
            rank_table_file = default_config['model']['rank_table_file']
            hccl_connect_time = default_config['model']['hccl_connect_time']
            os.environ['RANK_TABLE_FILE'] = rank_table_file
            os.environ['HCCL_CONNECT_TIME'] = hccl_connect_time

        self.config = MindFormerConfig(model_config)

        update_checkpoint_config(self.config)

        # build streamer
        self.tokenizer = get_tokenizer(self.config)
        self.streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True)

        process_list = []
        for i in range(self.device_num):
            input_q = Queue()
            output_q = Queue()
            self.input_q_list.append(input_q)
            self.output_q_list.append(output_q)
            device_id = self.device_id
            if self.device_num > 1:
                device_id = i + self.device_range[0]
            if i == 0:
                process_list.append(
                    Process(target=generate_process,
                            kwargs={'device_id': device_id,
                                    'device_num': self.device_num,
                                    'device_range': self.device_range,
                                    'config': self.config,
                                    'input_q': input_q,
                                    'output_q': output_q,
                                    'streamer': self.streamer}))
            else:
                process_list.append(
                    Process(target=generate_process,
                            kwargs={'device_id': device_id,
                                    'device_num': self.device_num,
                                    'device_range': self.device_range,
                                    'config': self.config,
                                    'input_q': input_q,
                                    'output_q': output_q}))
        for p in process_list:
            p.start()

        # model warm up
        warm_up_data = dict(inputs='你是谁？', max_length=10, do_sample=False)
        for input_q in self.input_q_list:
            input_q.put(warm_up_data)
        for output_q in self.output_q_list:
            output_q.get()
        logger.info("Model warm-up is finish. Ready to predict.")

    def infer(self,
              inputs: str,
              do_sample: bool,
              temperature: float,
              repetition_penalty: float,
              top_k: int,
              top_p: float,
              max_length: int,
              stream: bool):
        """Infer"""
        infer_data = {
            'inputs': inputs,
        }
        infer_data.update(do_sample=do_sample)
        infer_data.update(temperature=temperature)
        infer_data.update(repetition_penalty=repetition_penalty)
        infer_data.update(top_k=top_k)
        infer_data.update(top_p=top_p)
        infer_data.update(max_length=max_length)
        infer_data.update(stream=stream)

        if max_length > self.config.model.model_config.seq_length:
            raise ValueError("max_length cannot be larger than model seq_length")
        input_tokens = self.tokenizer(inputs)["input_ids"]
        if len(input_tokens) >= max_length:
            raise ValueError("input tokens length cannot be larger than max_length")

        self.streamer.clear()
        for input_q in self.input_q_list:
            input_q.put(infer_data)

    def get_res_iter(self):
        """Get result iteratively"""
        for text in self.streamer:
            yield text

    def get_res(self):
        """Get result"""
        result = None
        for output_q in self.output_q_list:
            result = output_q.get()
        return result
