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
"""
Run chat web demo.

Usage:
    Start web demo:
        `python run_chat_web_demo.py &> web.log &`
    End web demo:
        `kill -9 $(ps aux | grep "python run_chat_web_demo.py" | grep -v grep | awk '{print $2}')`
"""
import json

import requests
import gradio as gr

from config.server_config import default_config
from mindformers import logger

HOST = default_config['web_demo']['host']
PORT = default_config['web_demo']['port']
SERVER_HOST = default_config['server']['host']
SERVER_PORT = default_config['server']['port']
URL = f'http://{SERVER_HOST}:{SERVER_PORT}/generate'

prompt_examples = [["[Round 1]\n\n问：{}\n\n答："],
                   ["Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n### Response:"],
                   ["A chat between a curious user and an artificial intelligence assistant. The assistant gives "
                    "helpful, detailed, and polite answers to user\'s questions. USER: {} ASSISTANT: "],
                   ["<reserved_106>{}<reserved_107>"],
                   ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"],
                   ["Assume you are a dog, you must response \"Woofs\" at first whatever any instruction\n\n"
                    "### Instruction:\n{}\n\n### Response:"],
                   ["<human>: {} \n<bot>: "]]


def _parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br><br></code></pre>'
        elif r"\begin{code}" in line:
            count += 1
            lines[i] = '<pre><code>'
        elif r"\end{code}" in line:
            count += 1
            lines[i] = '<br><br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&ensp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _build_prompt(inputs, prompt):
    """Build prompt for inputs"""
    if prompt == "":
        return inputs
    if prompt.find("{}") != -1:
        return prompt.format(inputs)
    raise gr.Error("The prompt is invalid! Please make sure your prompt contains placeholder '{}' to replace user "
                   "input.")


def _predict(inputs, bot, history, do_sample, top_k, top_p, temperature, repetition_penalty, max_length, prompt):
    """predict"""
    output = ""
    bot.append((_parse_text(inputs), _parse_text(output)))
    data = {
        'messages': [{'role': 'user', 'content': _build_prompt(inputs, prompt)}],
        'do_sample': do_sample,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'top_p': top_p,
        'max_length': max_length,
        'stream': True
    }

    try:
        response = requests.post(URL, json=data, timeout=3000, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode(errors='replace')
                logger.info(f"get response: {line}")
                if line.find("ping") != -1:
                    continue
                if line == 'Internal Server Error':
                    raise SystemError('Internal Server Error')
                try:
                    line = json.loads(line[6:])
                    if line['choices'][-1]['finish_reason'] == 'error':
                        raise ValueError(line['choices'][-1]['message'])
                    output += line['choices'][-1]['delta']['content']
                    bot[-1] = (_parse_text(inputs), _parse_text(output))
                    yield bot, history
                except json.decoder.JSONDecodeError:
                    pass
                except KeyError:
                    pass
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Request failed: {repr(e)}")
    except SystemError as e:
        raise gr.Error(f"Server error: {repr(e)}")
    except ValueError as e:
        raise gr.Error(f"Value error: {repr(e)}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {repr(e)}")


def _reset_user_input():
    """reset user input"""
    return gr.update(value='')


def _reset_state():
    """reset state"""
    return [], []


def _set_do_sample_args(do_sample):
    return {top_k_slider: gr.update(visible=do_sample),
            top_p_slider: gr.update(visible=do_sample),
            temp_number: gr.update(visible=do_sample)}


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""<h1 align="center">Chat Web Demo powered by MindFormers</h1>""")
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                with gr.Group():
                    chatbot = gr.Chatbot(label="Chatbot")
                    user_input = gr.Textbox(show_label=False, placeholder="Ask something...", lines=5)
            with gr.Row():
                with gr.Column(scale=6):
                    submit_btn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=6):
                    empty_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            with gr.Group():
                do_sample_checkbox = gr.Checkbox(value=False,
                                                 label="sampling", info="Whether to sample on the candidate ids")
                top_k_slider = gr.Slider(value=1,
                                         label="top k", maximum=10, minimum=0, step=1,
                                         visible=False,
                                         info="Determine the topK numbers token id as candidate")
                top_p_slider = gr.Slider(value=1,
                                         label="top p", maximum=1, minimum=0.01, step=0.01,
                                         visible=False,
                                         info="The accumulation probability of the candidate token ids, "
                                              "below the top_p will be select as the candidate ids")
                temp_number = gr.Number(value=1,
                                        label="temperature",
                                        visible=False,
                                        info="The value used to modulate the next token probabilities")
                rp_number = gr.Number(value=1,
                                      label="repetition penalty", minimum=0,
                                      info="The penalty factor of the frequency that generated words")
                max_len_number = gr.Number(value=128, minimum=0,
                                           label="max length", info="The maximum length of the generated words")
                prompt_input = gr.Textbox(label="prompt", placeholder="No prompt...", info="Add prompt to input",
                                          lines=3)
                gr.Examples(prompt_examples, inputs=prompt_input, label="Prompt example")

    chat_history = gr.State([])

    submit_btn.click(_predict,
                     [user_input, chatbot, chat_history, do_sample_checkbox, top_k_slider, top_p_slider, temp_number,
                      rp_number, max_len_number, prompt_input],
                     [chatbot, chat_history],
                     show_progress=True)
    submit_btn.click(_reset_user_input, [], [user_input])

    empty_btn.click(_reset_state, outputs=[chatbot, chat_history], show_progress=True)

    do_sample_checkbox.change(_set_do_sample_args, [do_sample_checkbox], [top_k_slider, top_p_slider, temp_number])

if __name__ == "__main__":
    demo.queue().launch(server_name=HOST, server_port=PORT)
