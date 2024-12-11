import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from telechat2_for_autoawq import TeleChatV2AWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "YOUR_MODEL_PATH" #修改为模型路径
model = TeleChatV2AWQForCausalLM.from_pretrained(model_path,
                                                 model_type='telechat',
                                                 safetensors=False,
                                                 device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#默认的量化配置配置
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

import pandas as pd

sft_data = pd.read_json("YOUR_DATA_PATH",lines=True, dtype=str)
"""
格式:
[
    {
        "instuction":"instruct1",
        "output":"ouput1"
    },
    {
       "instuction":"instruct2",
       "output":"ouput2"
    }
]
"""

from tqdm import tqdm

data = []
for i in tqdm(range(len(sft_data))):
    instruction = sft_data.loc[i, "instruction"]
    output = sft_data.loc[i, "output"]
    text = instruction + output
    data.append(text.strip())

module = model.get_model_layers(model.model)[0]

#默认样本128条，512 length
model.quantize(tokenizer, quant_config=quant_config, calib_data=data, max_calib_samples=128, max_calib_seq_len=512)

from transformers import AwqConfig, AutoConfig

quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
).to_dict()

model.model.config.quantization_config = quantization_config

#量化后模型保存地址
quant_path = "YOUR_OUTPUT_PATH"

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)