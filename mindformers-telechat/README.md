# 欢迎来到MindSpore Transformers（MindFormers）

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindformers.svg)](https://pypi.org/project/mindformers)

## 一、介绍

MindSpore Transformers套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

MindSpore Transformers套件基于MindSpore内置的并行技术和组件化设计，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换；
- 提供灵活易用的个性化并行配置；
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略；
- 一键启动任意任务的单卡/多卡训练、微调、评估、推理流程；
- 支持用户进行组件化配置任意模块，如优化器、学习策略、网络组装等；
- 提供Trainer、pipeline、AutoClass等高阶易用性接口；
- 提供预置SOTA权重自动下载及加载功能；
- 支持人工智能计算中心无缝迁移部署；

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

- 📝 **[MindFormers文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)**
- 📝 [大模型低参微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/parameter_efficient_fine_tune.html)
- 📝 [AICC指导教程](docs/readthedocs/source_zh_cn/docs/practice/AICC.md)

### 支持模型

MindFormers已支持大部分模型的[LoRA微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/parameter_efficient_fine_tune.html)以及[LoRA权重合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html#lora权重合并)功能，具体可参考各模型文档启动模型的LoRA微调任务。

当前MindFormers支持的模型列表如下：

<table>
  <thead>
    <tr>
      <th> 模型 </th>
      <th> 参数 </th>
      <th> 序列 </th>
      <th> 预训练 </th>
      <th> 微调 </th>
      <th> 推理 </th>
      <th> <a href="docs/feature_cards/Pet_Tuners.md"> LoRA </a> </th>
      <th> 对话 </th>
      <th> 评估 </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"> <a href="docs/model_cards/llama2.md"> LLaMA2 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/llama2/run_llama2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/llama2/run_llama2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/llama2/run_llama2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/llama3/llama3.md"> LLaMA3 </a> </td>
      <td> 8B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/llama3/run_llama3_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 8K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/llama3/run_llama3_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
<tbody>
    <tr>
      <td rowspan="2"> <a href="research/llama3_1/llama3_1.md"> LLaMA3.1 </a> </td>
      <td> 8B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/llama3_1/llama3_1.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/llama3_1/llama3_1.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/baichuan2/baichuan2.md"> Baichuan2 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/baichuan2/run_baichuan2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/baichuan2/run_baichuan2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/glm2.md"> GLM2 </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/glm2/run_glm2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL / Rouge </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/glm3.md"> GLM3 </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/glm3/run_glm3_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/glm3.md"> GLM3-32K </a> </td>
      <td> 6B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/glm32k/run_glm32k_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/glm4.md"> GLM4 </a> </td>
      <td> 9B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/glm4/run_glm4_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/cogvlm2_video.md"> CogVLM2-Video </a> </td>
      <td> 13B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="docs/model_cards/cogvlm2_video.md"> docs </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/cogvlm2_image.md"> CogVLM2-Image </a> </td>
      <td> 19B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="docs/model_cards/cogvlm2_image.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/qwen/qwen.md"> Qwen </a> </td>
      <td> 7B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen/qwen.md"> docs </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> C-Eval </td>
    </tr>
    <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen/qwen.md"> docs </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> C-Eval </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="7"> <a href="research/qwen1_5/qwen1_5.md"> Qwen1.5 </a> </td>
      <td> 0.5B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 7B </td>
      <td> 32K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 14B </td>
      <td> 32K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 72B </td>
      <td> 32K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen1_5/qwen1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="6"> <a href="research/qwen2/qwen2.md"> Qwen2 </a> </td>
      <td> 0.5B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 1.5B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 7B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 57B-A14B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 57B </td>
      <td> 32K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 72B </td>
      <td> 128K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/qwen2/qwen2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="research/qwenvl/qwenvl.md"> QwenVL </a> </td>
      <td> 9.6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/qwenvl/run_qwenvl_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/internlm/internlm.md"> InternLM </a> </td>
      <td> 7B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/internlm/run_internlm_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
    <tr>
      <td> 20B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/internlm/run_internlm_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/internlm2/internlm2.md"> InternLM2 </a> </td>
      <td> 7B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/internlm2/run_internlm2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 20B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td> <a href="scripts/examples/internlm2/run_internlm2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="research/yi/yi.md"> Yi </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/yi/run_yi_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 34B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/yi/run_yi_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="research/mixtral/mixtral.md"> Mixtral </a> </td>
      <td> 8x7B </td>
      <td> 32K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/mixtral/mixtral.md"> docs </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="research/deepseek/deepseek.md"> DeepSeek Coder </a> </td>
      <td> 33B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/deepseek/deepseek.md"> docs </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="research/deepseek1_5/deepseek1_5.md"> DeepSeek Coder1.5 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/deepseek1_5/deepseek1_5.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="research/deepseek2/deepseek2.md"> DeepSeekV2 </a> </td>
      <td> 236B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> <a href="research/deepseek2/deepseek2.md"> docs </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/codellama.md"> CodeLlama </a> </td>
      <td> 34B </td>
      <td> 4K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/codellama/run_codellama_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> HumanEval </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/gpt2.md"> GPT2 </a> </td>
      <td> 13B </td>
      <td> 2K </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td> <a href="scripts/examples/gpt2/run_gpt2_predict.sh"> generate </a> </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> PPL </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="docs/model_cards/whisper.md"> Whisper </a> </td>
      <td> 1.5B </td>
      <td> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> &#x2713 </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
</table>

## 二、安装

### 版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.10。

| MindFormers | MindPet | MindSpore | CANN |                                  驱动固件                                  | 镜像链接 |  备注  |
|:-----------:|:-------:|:---------:|:----:|:----------------------------------------------------------------------:|:----:|:----:|
|    1.3.0    |  1.0.4  |   2.4.0   |  -   | [driver](https://www.hiascend.com/hardware/firmware-drivers/community) |  -   | 版本分支 |

当前MindFormers建议使用如上的软件配套关系。其中CANN和固件驱动的安装需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本。

#### 兼容性说明

MindFormers与MindSpore有如下兼容关系：

| MindFormers | MindSpore | 兼容性 |
|:-----------:|:---------:|:---:|
|    1.3.0    |    2.3    |  √  |
|    1.2.0    |    2.4    |  √  |

### 源码编译安装

MindFormers目前支持源码编译安装，用户可以执行如下命令进行安装。

```shell
git clone -b r1.3.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、使用指南

MindFormers支持模型启动预训练、微调、推理、评测等功能，可点击[支持模型](#支持模型)中模型名称查看文档完成上述任务，以下为模型分布式启动方式的说明与示例。

MindFormers推荐使用分布式方式拉起模型训练、推理等功能，目前提供`scripts/msrun_launcher.sh`分布式启动脚本作为模型的主要启动方式，`msrun`特性说明可以参考[msrun启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)。
该脚本主要输入参数说明如下：

  | **参数**           | **单机是否必选** | **多机是否必选** |     **默认值**      | **说明**           |
  |------------------|:----------:|:----------:|:----------------:|------------------|
  | WORKER_NUM       |  &check;   |  &check;   |        8         | 所有节点中使用计算卡的总数    |
  | LOCAL_WORKER     |     -      |  &check;   |        8         | 当前节点中使用计算卡的数量    |
  | MASTER_ADDR      |     -      |  &check;   |    127.0.0.1     | 指定分布式启动主节点的ip    |
  | MASTER_PORT      |     -      |  &check;   |       8118       | 指定分布式启动绑定的端口号    |
  | NODE_RANK        |     -      |  &check;   |        0         | 指定当前节点的rank id   |
  | LOG_DIR          |     -      |  &check;   | output/msrun_log | 日志输出路径，若不存在则递归创建 |
  | JOIN             |     -      |  &check;   |      False       | 是否等待所有分布式进程退出    |
  | CLUSTER_TIME_OUT |     -      |  &check;   |       7200       | 分布式启动的等待时间，单位为秒  |

> 注：如果需要指定`device_id`启动，可以设置环境变量`ASCEND_RT_VISIBLE_DEVICES`，如要配置使用2、3卡则输入`export ASCEND_RT_VISIBLE_DEVICES=2,3`。

### 单机多卡

```shell
# 1. 单机多卡快速启动方式，默认8卡启动
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}"

# 2. 单机多卡快速启动方式，仅设置使用卡数即可
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" WORKER_NUM

# 3. 单机多卡自定义启动方式
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" \
  WORKER_NUM MASTER_PORT LOG_DIR JOIN CLUSTER_TIME_OUT
 ```

- 使用示例

  ```shell
  # 单机多卡快速启动方式，默认8卡启动
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune"

  # 单机多卡快速启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" 8

  # 单机多卡自定义启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" \
    8 8118 output/msrun_log False 300
  ```

### 多机多卡

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，
所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

  ```shell
  # 多机多卡自定义启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config {CONFIG_PATH} \
   --run_mode {train/finetune/eval/predict}" \
   WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK LOG_DIR JOIN CLUSTER_TIME_OUT
  ```

- 使用示例

  ```shell
  # 节点0，节点ip为192.168.1.1，作为主节点，总共8卡且每个节点4卡
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 0 output/msrun_log False 300

  # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 1 output/msrun_log False 300
  ```

### 单卡启动

MindFormers提供`run_mindformer.py`脚本作为单卡启动方法，该脚本可以根据模型配置文件，完成支持模型的单卡训练、微调、评估、推理流程。

```shell
# 运行run_mindformer.py的入参会覆盖模型配置文件中的参数
python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
```

## 四、贡献

欢迎参与社区贡献，可参考MindSpore贡献要求[Contributor Wiki](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/mindformers_contribution.html)。

## 五、许可证

[Apache 2.0许可证](LICENSE)
