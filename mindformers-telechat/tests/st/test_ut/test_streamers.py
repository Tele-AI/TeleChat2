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
"""test generation.streamer schedule."""
from threading import Thread
from mindformers import GPT2LMHeadModel, GPT2Tokenizer, TextStreamer, TextIteratorStreamer


def test_text_streamer_schedule():
    """
    Feature: TextStreamer Schedule
    Description: Test TextStreamer Generation.Streamer Schedule
    Expectation: ValueError
    """
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

    streamer = TextStreamer(tok)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    result = model.generate(inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
    result = result[0].tolist()

    assert result == \
           [2025, 3649, 8379, 25, 530, 11, 734, 11, 1115, 11, 1440, 11, 1936, 11, 2237, 11, 3598, 11, 3624, 11]


def test_text_iterator_streamer_schedule():
    """
    Feature: TextIteratorStreamer Schedule
    Description: Test TextIteratorStreamer Generation.Streamer Schedule
    Expectation: ValueError
    """
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

    streamer = TextIteratorStreamer(tok)

    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(input_ids=inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text = "".join([generated_text, new_text])

    assert generated_text == "An increasing sequence: one, two, three, four, five, six, seven, eight,"


def test_text_iterator_streamer_schedule_batch():
    """
    Feature: Batch Streamer Schedule
    Description: Test TextIteratorStreamer Generation.Streamer Schedule, with batch input.
    Expectation: ValueError
    """
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    text_batch = ["An increasing sequence: one,",
                  "The highest mountain in the world is",
                  "The largest river in China is"]
    inputs = tok(text_batch, max_length=8, padding='max_length', return_tensors=None, add_special_tokens=False)

    streamer = TextIteratorStreamer(tok)

    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(input_ids=inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    batch_size = len(text_batch)
    generated_text = [""] * batch_size
    for new_text in streamer:
        for i in range(batch_size):
            generated_text[i] = "".join([generated_text[i], new_text[i]])

    assert generated_text == \
           ['An increasing sequence: one, two, three, four, five, six, seven, eight,',
            'The highest mountain in the world is the Himalayas, which is the highest mountain in the world',
            'The largest river in China is the Yangtze River, which flows through the heart of the country']
