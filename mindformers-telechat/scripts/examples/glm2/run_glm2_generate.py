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
"""glm2 predict example."""
import os
import argparse

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2Tokenizer, ChatGLM2ForConditionalGeneration


def main(config_path, load_checkpoint):
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
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_name)

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

    queries = ["你好",
               "请介绍一下杭州",
               "那里有什么好吃的吗"]
    history = []
    for query in queries:
        prompt = tokenizer.build_prompt(query, history=history)
        input_ids = tokenizer(prompt)["input_ids"]

        output = network.generate([input_ids],
                                  max_length=model_config.seq_length,
                                  do_sample=False,
                                  top_p=3,
                                  top_k=0.7,
                                  temperature=1)

        output = output[0][len(input_ids):]
        response = tokenizer.decode(output)
        print(response)
        history += [(query, response)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_glm2_6b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.load_checkpoint
    )

# 推理结果：
# response1:
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
#
# response2:
# 杭州是中国浙江省省会，位于浙江省东南部，地处浙江省北部，东临东海，南接福建省，北与江苏省毗邻，是中国著名的旅游城市之一。
#
# 杭州有着悠久的历史和文化，被誉为“人间天堂”，被誉为“南宋都城”，是中国南方著名的历史文化名城之一。杭州还被誉为“全国最具幸福感城市”，具有丰富的历史遗存、优美的自然风光和浓郁的文化氛围。
#
# 杭州的经济以服务业为主导产业，特别是交通运输、仓储和邮政业。同时，杭州也是中国重要的电子商务和互联网产业基地之一，被誉为“中国电子商务之都”。
#
# 杭州的著名景点包括西湖、灵隐寺、千岛湖、钱塘江等。西湖是中国著名的风景名胜区之一，被誉为“人间天堂”，灵隐寺是中国著名的佛教寺庙之一，千岛湖和钱塘江是中国著名的自然风景区之一。
#
# 杭州还拥有丰富的人文资源，被誉为“人间天堂”的杭州西湖、灵隐寺、千岛湖、钱塘江等景点，以及宋城、南宋御街等历史文化景点，都是游客前来杭州旅游的热门景点。
#
# response3:
# 杭州是中国著名的美食城市之一，有许多特色美食和传统菜肴。以下是一些杭州的著名美食:
#
# 1. 西湖醋鱼：这是杭州最著名的菜肴之一，鱼肉鲜美，入口即化，佐以香醋、糖、姜丝等调料，口感酸甜适中。
#
# 2. 龙井虾仁：以当地特产的龙井茶为佐料，将鲜嫩的虾仁炒制而成，口感清香可口。
#
# 3. 灌汤包：又称小笼包，是杭州的传统点心之一。包子的皮软馅鲜，汤汁鲜美，非常受欢迎。
#
# 4. 姜母鸭：这是一道杭帮菜，以鸭肉、姜母、葱等调料烹制而成，口感鲜美。
#
# 5. 老字号小吃：杭州还有很多老字号小吃店，如胡同口烤肉串、孔府家宴、宋嫂鱼羹等，是当地居民和游客的美食选择。
#
# 此外，杭州还有许多特色小吃，如粽子、臭豆腐、糯米鸡、肉夹馍、鸭血粉丝汤等，让人垂涎欲滴。
