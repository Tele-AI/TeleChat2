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
"""Baichuan13b Train/Finetune/Eval/Predict scripts."""

import argparse

import mindspore as ms

from mindformers import LlamaConfig, LlamaTokenizer, TextStreamer

from baichuan_13b import Baichuan13BForCausalLM


USER_TOKEN_ID = 195
ASSISTANT_TOKEN_ID = 196


def main(tk_config='./', ckpt=None, max_new_tokens=512):
    """main function."""

    # initialize Graph Mode
    ms.set_context(mode=0)

    tokenizer = LlamaTokenizer.from_pretrained(tk_config)

    config = LlamaConfig(batch_size=1,  # add for increase predict
                         seq_length=1024,
                         hidden_size=5120,
                         num_layers=40,
                         num_heads=40,
                         vocab_size=64000,
                         multiple_of=107,
                         rms_norm_eps=1.0e-6,
                         bos_token_id=1,
                         eos_token_id=2,
                         pad_token_id=0,
                         ignore_token_id=-100,
                         use_past=True,
                         repetition_penalty=1.1,
                         temperature=0.3,
                         max_decode_length=1024,
                         top_k=5,
                         top_p=0.85,
                         do_sample=True,
                         checkpoint_name_or_path=ckpt)

    baichuan_13b = Baichuan13BForCausalLM(config)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    messages = []
    while True:
        messages.append({"role": "user", "content": input("请输入：")})
        input_ids = build_chat_input(config, tokenizer, messages, max_new_tokens)
        outputs = baichuan_13b.generate(input_ids,
                                        streamer=streamer,
                                        temperature=0.3,
                                        top_k=5,
                                        top_p=0.85,
                                        repetition_penalty=1.1,
                                        do_sample=True)

        response = tokenizer.decode(outputs[0][len(input_ids):], skip_speical_tokens=True)
        messages.append({"role": "assistant", "content": response})


def build_chat_input(config, tokenizer, messages, max_new_tokens=None):
    """add prompt for baichuan input, and truncate input if too long."""
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        r = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                if i != 0:
                    raise ValueError(f"i should be 0,but got {i}")
                system = message["content"]
                continue
            if message["role"] == split_role and r:
                rounds.append(r)
                r = []
            r.append(message)
        if r:
            rounds.append(r)
        return system, rounds

    max_new_tokens = max_new_tokens or config.max_decode_length // 2
    max_input_tokens = config.max_decode_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if message["role"] == "user":
                round_tokens.append(USER_TOKEN_ID)
            else:
                round_tokens.append(ASSISTANT_TOKEN_ID)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if not history_tokens or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(ASSISTANT_TOKEN_ID)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=None, type=str,
                        help='config used to init tokenizer.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--max_new_tokens', default=None, type=int,
                        help='max new tokens will be generated.')
    args = parser.parse_args()

    main(tk_config=args.config, ckpt=args.load_checkpoint, max_new_tokens=args.max_new_tokens)
