# KnowLM

KnowLM是一个知识增强的开源语言大模型框架，旨在提供灵活且可定制的工具集和并发布相应的模型，帮助研究人员和开发者更好地处理大模型知识更新和知识谬误等问题，具体包括：

1.**知识提示**：基于知识提示技术从知识图谱生成和优化指令数据以解决知识抽取问题

2.**知识编辑**：基于知识编辑技术对齐大模型内过时及价值观不正确的知识以解决知识谬误问题

3.**知识交互**：基于知识交互技术实现工具组合学习及多智能体协作以解决语言模型具身认知问题

现阶段KnowLM已发布基于LLaMA1的13B基础模型一个（KnowLM-13B-Base），知识抽取大模型一个（KnowLM-13B-ZhiXi，KnowLM-13B-IE2个版本）。

项目主页：[KnowLM](https://github.com/zjunlp/KnowLM)

## KnowLM-13B-ZhiXi

KnowLM-13B-Base以 LlaMA-13B 为基础，使用中英文双语数据进行了二次预训练，提高了模型对中文的理解能力。KnowLM-13B-ZhiXi在 Knowlm-13B-Base 的基础上，利用知识图谱转换指令技术生成数据对该模型进行了微调。详情请参考[KnowLM](https://github.com/zjunlp/KnowLM)项目

```text
@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM: An Open-sourced Knowledgeable Large Language Model Framework},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}

@article{wang2023easyedit,
  title={EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}
```

## 快速使用

### KnowLM-13B-ZhiXi 预训练权重转换

从huggingface下载[KnowLM-13B-ZhiXi](https://huggingface.co/zjunlp/knowlm-13b-zhixi/tree/main);把文件全部下载下来

执行权重转换脚本

```shell
python research/knowlm/convert_weight.py --torch_bin_path TORCH_BIN_PATH --mindspore_ckpt_path MS_CKPT_PATH
```

```text
# 参数说明
TORCH_BIN_PATH: huggingface权重保存目录下任意权重bin文件，根据该文件路径读取目录下所有权重
MS_CKPT_PATH: mindspore权重保存文件路径
```

### API方式调用

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
> 遵从Knowlm-13B-zhixi的license，本模型需要用户自行下载权重进行处理

- pipeline接口开启快速推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_target="Ascend")
# init knowlm-13b-zhixi model
knowlm_model_path = "/path/to/your/weight.ckpt" # knowlm-13B-zhixi ckpt path
knowlm_config = LlamaConfig(
    seq_length=2048,
    vocab_size=32000,
    pad_token_id=0,
    checkpoint_name_or_path=knowlm_model_path,
    hidden_size=5120,
    num_layers=40,
    num_heads=40,
    rms_norm_eps=1e-6
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
peline_result = pipeline_task("你非常了解一些健康生活的习惯，请列举几个健康生活的建议", top_k=3, do_sample=True, top_p=0.95, repetition_penalty=1.3, max_length=256)

print(peline_result)
#你非常了解一些健康生活的习惯，请列举几个健康生活的建议：1.每天坚持锻炼30分钟以上。 2.不吸烟，不酗酒。 3.少吃高脂肪食物。 4.多吃蔬菜和水果。 5.保证充足的睡眠。 6.保持良好的心情。 7.定期体检。 8.养成良好的卫生习惯
```

### KnowLM-13B-ZhiXi Lora微调训练

#### 前期准备

环境要求和微调准备参考[llama-7b-lora的前期准备](https://gitee.com/rolnan_f/mindformers/blob/dev/docs/model_cards/llama.md#%E5%89%8D%E6%9C%9F%E5%87%86%E5%A4%87)

#### 数据集准备

微调训练采用的数据集为alpaca数据集，数据处理部分可以参考[llama-7b的数据处理过程](https://gitee.com/rolnan_f/mindformers/blob/dev/docs/model_cards/llama.md#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87-%E5%BE%AE%E8%B0%83)

给出了knowlm-13b-zhixi适配的lora配置文件-run_knowlm_13b.yaml

#### 脚本启动

```sh
cd scripts
# 单卡启动
bash run_standalone.sh run_knowlm_13b.yaml [DEVICE_ID] finetune
# 多卡启动（以单机八卡为例）
bash run_distribute.sh [RANK_TABLE_FILE] run_knowlm_13b.yaml [0,8] finetune
```

### 训练速度和精度

我们在华为昇腾NPUAscend 910 32GB显存上进行训练，采用了fp32（单精度浮点数）的数据格式进行计算。在每个step中，Lora所需的时间约为2480ms，同时，每秒处理的样本数是0.81samples s/p

在不同数据集上的精度如下
|f1|A800-3epoch|v100-1.6epoch|Ascend-3epoch|
|-|-|-|-|
|GIDS|68.64|74.04|76.23|
|NYT11|72.43|75.51|75.14|
|SciERC|25.15|37.28|36.49|
|kbp37|93.44|95.48|95.73|

