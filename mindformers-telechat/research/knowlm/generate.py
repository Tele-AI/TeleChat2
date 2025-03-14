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
'''Knowlm weight convert'''
import yaml
from mindspore import context
from mindformers.tools.check_rules import check_yaml_depth_before_loading
from mindformers.pipeline import pipeline
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_target="Ascend")
with open("./knowlm.yaml", 'r') as file:
    check_yaml_depth_before_loading(file)
    file.seek(0)
    knowlm_data = yaml.load(file, Loader=yaml.SafeLoader)

# init knowlm-13b-zhixi model
knowlm_model_path = "/path/to/your/weight.ckpt" # knowlm-13B-zhixi ckpt path
knowlm_config = LlamaConfig(
    seq_length=knowlm_data['model_config']['seq_length'],
    vocab_size=knowlm_data['model_config']['vocab_size'],
    pad_token_id=knowlm_data['model_config']['pad_token_id'],
    checkpoint_name_or_path=knowlm_data['model_config']['checkpoint_name_or_path'],
    hidden_size=knowlm_data['model_config']['hidden_size'],
    num_layers=knowlm_data['model_config']['num_layers'],
    num_heads=knowlm_data['model_config']['num_heads'],
    rms_norm_eps=knowlm_data['model_config']['rms_norm_eps']
)
knowlm_model = LlamaForCausalLM(
    config=knowlm_config
)
# init knowlm-13b-zhixi tokenizer
tokenizer_path = "/path/to/your/tokenizer" # knowlm-13B-zhixi tokenizer.model path
tokenizer = LlamaTokenizer(
    vocab_file=tokenizer_path
)
pipeline_task = pipeline("text_generation", model=knowlm_model, tokenizer=tokenizer, max_length=32)
peline_result = pipeline_task(
    "你非常了解一些健康生活的习惯，请列举几个健康生活的建议",
    top_k=3,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.3,
    max_length=256)

print(peline_result)
