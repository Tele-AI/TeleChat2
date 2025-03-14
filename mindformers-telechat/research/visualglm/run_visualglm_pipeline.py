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
"""visualglm pipeline runner."""
import argparse

import mindspore as ms
from mindspore.dataset import vision
from mindspore.dataset.vision.utils import Inter

from mindformers.pipeline import pipeline
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger

from visualglm import VisualGLMImageToTextGeneration
from visualglm_config import VisualGLMConfig
from visualglm_processor import VisualGLMProcessor


def init_context(device_id):
    """ init context """
    ms.set_context(mode=0, device_target="Ascend", device_id=device_id, max_device_memory="59GB")  # Ascend, CPU


def main(args):
    init_context(device_id=args.device_id)
    model_config = VisualGLMConfig.from_pretrained(args.config_path)
    model_config.max_txt_len = args.seq_length

    if args.checkpoint is not None:
        logger.info(f"checkpoint: {args.checkpoint}")
        model_config.checkpoint_name_or_path = args.checkpoint

    if args.batch_size > 1:
        model_config.batch_size = args.batch_size
    else:
        model_config.batch_size = 1

    model_config.text_config.batch_size = model_config.batch_size
    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    model = VisualGLMImageToTextGeneration(model_config)
    processor = VisualGLMProcessor.from_pretrained(args.config_path)
    processor.image_processor.resize.resize = vision.transforms.Resize((224, 224), Inter.BICUBIC)
    tokenizer = processor.tokenizer

    logger.info(f"batch_size is {model_config.batch_size}")

    pipeline_task = pipeline(task='visualglm_image_to_text_generation', model=model,
                             image_processor=processor.image_processor,
                             tokenizer=tokenizer, batch_size=model_config.batch_size)

    predict_result = pipeline_task({
        "image": args.image_path,
        "prompt": args.prompt})
    logger.info(predict_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="visualglm_6b", type=str, required=False, help='model type')
    parser.add_argument('--config_path', default="run_visualglm_6b_image_to_text_generation.yaml",
                        type=str, required=False, help='config path')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='device id')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch_size')
    parser.add_argument('--checkpoint', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--generate_repeat_time', type=int, default=1, required=False, help='generate repeat time')
    parser.add_argument('--use_past', type=str2bool, default=True, required=False, help='whether use past')
    parser.add_argument('--do_sample', type=str2bool, default=False, required=False, help='whether do sample')
    parser.add_argument('--top_p', type=float, default=1, required=False, help='top p')
    parser.add_argument('--top_k', type=int, default=0, required=False, help='top k')
    parser.add_argument('--seq_length', type=int, default=32, required=False, help='seq length')
    parser.add_argument('--image_path', type=str, default=None, required=False, help='image path')
    parser.add_argument('--prompt', type=str, default=None, required=False, help='')
    args_ = parser.parse_args()
    print(args_)
    main(args_)
