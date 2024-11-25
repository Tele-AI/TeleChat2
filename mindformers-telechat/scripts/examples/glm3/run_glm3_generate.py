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
"""glm3 predict example."""
import os
import argparse
from copy import deepcopy

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration, ChatGLM3Tokenizer


def get_model(config_path, load_checkpoint):
    """build model for prediction."""
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = False
    config.load_checkpoint = load_checkpoint

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = ChatGLM2Config(**config.model.model_config)
    model_config.seq_length = 1024
    model_config.checkpoint_name_or_path = None
    model_name = config.trainer.model_name

    # build tokenizer
    tokenizer = ChatGLM3Tokenizer.from_pretrained(model_name)

    # build model
    network = ChatGLM2ForConditionalGeneration(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        # set auto transform ckpt
        if os.path.isdir(config.load_checkpoint):
            config.auto_trans_ckpt = True
        else:
            config.auto_trans_ckpt = False
        input_ids = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    return network, model_config, tokenizer


def main(config_path, load_checkpoint):
    model, model_config, tokenizer = get_model(config_path, load_checkpoint)

    queries = ["你好",
               "请介绍一下华为"]
    for query in queries:
        input_ids = tokenizer.build_chat_input(query, history=[], role='user')["input_ids"]
        outputs = model.generate(input_ids,
                                 max_length=model_config.seq_length,
                                 do_sample=False,
                                 top_k=1)
        for i, output in enumerate(outputs):
            output = output[len(input_ids[i]):]
            response = tokenizer.decode(output)
            print(response)

    # answer 1:
    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    # answer 2:
    # 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、
    # 云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。
    # 华为的主要业务包括电信网络设备、智能手机、电脑和消费电子产品。公司在全球范围内有超过190,000名员工,
    # 其中约一半以上从事研发工作。华为以其高品质的产品和服务赢得了全球客户的信任和好评,也曾因其领先技术和创新精神而获得多项国际奖项和认可。
    # 然而,华为也面临着来自一些国家政府的安全问题和政治压力,其中包括美国政府对其产品的禁令和限制。
    # 华为一直坚称自己的产品是安全的,并采取了一系列措施来确保其产品的安全性和透明度。


def process_response(output, history):
    """process predict results."""
    content_dict = None
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content_dict = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                # pylint: disable=eval-used
                parameters = eval(content)
                content_dict = {"name": metadata.strip(), "parameters": parameters}
            else:
                content_dict = {"name": metadata.strip(), "content": content}
    return content_dict, history


def multi_role_predict(config_path, load_checkpoint):
    """multi-role predict process."""
    model, model_config, tokenizer = get_model(config_path, load_checkpoint)

    generate_config = {
        "max_length": model_config.seq_length,
        "num_beams": 1,
        "do_sample": False,
        "top_p": 1,
        "top_k": 1,
        "temperature": 1
    }

    # first input
    role = "system"
    text = "假设你现在是一个导游，请尽可能贴近这个角色回答问题。"
    history = []
    inputs = tokenizer.build_chat_input(text, history=history, role=role)['input_ids']
    history.append({'role': role, 'content': text})

    outputs = model.generate(inputs, **generate_config)
    outputs = outputs[0][len(inputs[0]):-1]
    response = tokenizer.decode(outputs)
    print(response)
    # 您好，我是您的人工智能助手，也可以是你的导游。请问有什么问题我可以帮您解答呢？
    response, history = process_response(response, history)
    print(f'history: {history}')

    # second input
    role = "user"
    text = "我打算1月份去海南玩，可以介绍一下海南有哪些好玩的，好吃的么？"
    inputs = tokenizer.build_chat_input(text, history=history, role=role)['input_ids']
    history.append({'role': role, 'content': text})
    outputs = model.generate(inputs, **generate_config)
    outputs = outputs[0][len(inputs[0]):-1]
    response = tokenizer.decode(outputs)
    print(response)
    # 当然可以！海南是一个风景优美、气候宜人的热带海洋省份，拥有丰富的旅游资源和美食。以下是一些您可能会感兴趣的景点和美食：
    # 1.
    # 景点：
    # - 海南岛：这是海南最著名的景点之一，拥有美丽的沙滩和热带雨林。
    # - 亚龙湾：这是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水。
    # - 南山寺：这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。
    # - 博鳌亚洲论坛永久会址：这是中国最著名的国际会议中心，也是亚洲地区最重要的政治、经济、文化论坛之一。
    # 2.
    # 美食：
    # - 海南鸡饭：这是海南最著名的美食之一，以鸡肉、米饭和椰汁为主要材料，味道鲜美。
    # - 海鲜：海南的海鲜非常新鲜，您可以在当地的海鲜市场或餐厅品尝到各种海鲜美食，如清蒸海鲜、红烧海鲜等。
    # - 椰子饭：这是海南最著名的传统美食之一，以椰子肉、糯米和椰子汁为主要材料，味道香甜。
    # - 海南粉：这是海南最著名的传统小吃之一，以米粉、猪肉、花生、蔬菜等为主要材料，味道鲜美。
    # 希望这些信息对您有所帮助，如果您还有其他问题，请随时问我。

    # third input
    response, history = process_response(response, history)
    role = "user"
    text = "哪里适合冲浪和潜水呢？"
    inputs = tokenizer.build_chat_input(text, history=history, role=role)['input_ids']
    history.append({'role': role, 'content': text})
    outputs = model.generate(inputs, **generate_config)
    outputs = outputs[0][len(inputs[0]):-1]
    response = tokenizer.decode(outputs)
    print(response)
    # 在海南，冲浪和潜水的好去处有很多。以下是一些建议：
    # 1.
    # 冲浪：
    # - 莺歌海：位于海南岛西海岸，是冲浪爱好者的天堂。这里的海浪适中，沙滩漂亮，非常适合冲浪。
    # - 三亚：位于海南岛南端，是海南最著名的冲浪胜地之一。这里的沙滩细腻，海浪较大，非常适合冲浪。
    # 2.
    # 潜水：
    # - 蜈支洲岛：位于海南岛东海岸，是海南最著名的潜水胜地之一。这里的潜水条件较好，能见度较高，水下生物丰富，非常适合潜水。
    # - 西沙群岛：位于海南岛东南方向，是海南另一个著名的潜水胜地。这里的潜水条件非常好，水下世界丰富多彩，非常适合潜水爱好者。
    # 当然，冲浪和潜水都需要一定的技能和经验，如果您是初学者，建议在专业人士的指导下进行。希望这些信息对您有所帮助，如果您还有其他问题，
    # 请随时问我。

    # forth input
    role = "user"
    text = "可以帮我做一份旅游攻略吗？"
    inputs = tokenizer.build_chat_input(text, history=history, role=role)['input_ids']
    history.append({'role': role, 'content': text})
    outputs = model.generate(inputs, **generate_config)
    outputs = outputs[0][len(inputs[0]):-1]
    response = tokenizer.decode(outputs)
    print(response)
    # 当然可以！以下是一份针对海南冲浪和潜水景点的旅游攻略：
    # 第一天：
    # 上午：抵达三亚，前往亚龙湾。在这里，您可以尽情享受阳光、沙滩和海水。亚龙湾是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水，
    # 非常适合冲浪和潜水。
    # 中午：在亚龙湾海滩附近享用午餐，品尝当地美食。
    # 下午：前往南山寺，这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。在这里，您可以领略到中国传统文化的魅力。
    # 晚上：返回三亚市区，在三亚湾海滩附近享用晚餐，并欣赏三亚夜景。
    # 第二天：
    # 上午：前往蜈支洲岛，这是海南最著名的潜水胜地之一。在这里，您可以尽情享受潜水的乐趣。
    # 中午：在蜈支洲岛附近享用午餐，品尝当地美食。
    # 下午：在蜈支洲岛潜水，欣赏美丽的海底世界。
    # 晚上：返回三亚市区，在三亚湾海滩附近享用晚餐，并欣赏三亚夜景。
    # 第三天：
    # 上午：前往博鳌亚洲论坛永久会址，这是中国最著名的国际会议中心，也是亚洲地区最重要的政治、经济、文化论坛之一。
    # 中午：在博鳌亚洲论坛永久会址附近享用午餐，品尝当地美食。
    # 下午：在博鳌亚洲论坛永久会址附近参观游览。
    # 晚上：返回三亚市区，在三亚湾海滩附近享用晚餐，并欣赏三亚夜景。
    # 第四天：
    # 上午：前往南山寺，这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。在这里，您可以领略到中国传统文化的魅力。
    # 中午：在南山寺附近享用午餐，品尝当地美食。
    # 下午：前往海南岛，这是海南最著名的景点之一，拥有美丽的沙滩和热带雨林。
    # 晚上：返回三亚市区，在三亚湾海滩附近享用晚餐，并欣赏三亚夜景。
    # 第五天：
    # 上午：前往亚龙湾，在这里，您可以尽情享受阳光、沙滩和海水。亚龙湾是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水，非常适合冲浪和潜水。
    # 中午：在亚龙湾海滩附近享用午餐，品尝当地美食。
    # 下午：前往蜈支洲岛，这是海南最著名的潜水胜地之一。在这里，您可以尽情享受潜水的乐趣。
    # 晚上：返回三亚市区，在三亚湾海滩附近享用晚餐，并欣赏三亚夜景。
    # 希望这份旅游攻略对您有所帮助，祝您旅途愉快！


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_glm3_6b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--multi_role', action='store_true',
                        help='if run model prediction in multi_role mode.')

    args = parser.parse_args()
    if args.multi_role:
        multi_role_predict(
            args.config_path,
            args.load_checkpoint
        )
    else:
        main(
            args.config_path,
            args.load_checkpoint
        )
