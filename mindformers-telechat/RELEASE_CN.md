# MindSpore Transformers Release Notes

## MindSpore Transformers 1.3.0 Release Notes

以下为 MindSpore Transformers (以下称为 MindFormers ) 套件 1.3.0 版本的变更日志，相较于1.2.0版本有以下关键新特性和 bugfix 。

### 新特性

- [安装验证](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/mindformers/mindformers.run_check.html)：新增了便捷的API用以查询MindFormers的安装是否成功。
- [日志优化]：优化 MindFormers 日志，打印信息更全面，更易于精度定位以及训练状态的监控。
- [LLM对话API](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/generation/mindformers.generation.GenerationMixin.html#mindformers.generation.GenerationMixin.chat)：提供了大型语言模型的对话文本生成推理接口。
- [量化推理](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/quantization.html#)：集成 MindSpore Golden Stick 工具组件，提供统一量化推理流程。
- [BIN格式数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/dataset.html#bin-%E6%A0%BC%E5%BC%8F%E6%95%B0%E6%8D%AE%E9%9B%86)：新增对 BIN 格式数据集的处理能力，包括如何制作 BIN 格式数据集和在任务中使用 BIN 格式数据集。
- [在线数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/dataset.html#%E5%9C%A8%E7%BA%BF%E6%95%B0%E6%8D%AE%E9%9B%86)：训练时支持加载在线数据集，无需本地离线处理。
- [榜单评测](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/evaluation.html)：基于 Harness 评测框架，支持加载 MindFormers 模型进行评测，支持自定义 prompt 和评测指标，包含 loglikelihood、 generate_until、 loglikelihood_rolling 三种类型的评测任务。基于 VLMEvalKit 评测框架，支持加载 MindFormers 多模态大模型进行评测，支持自定义 prompt 和评测指标，包含 MME、 MMEBench、 COCO caption 三种图文理解评估方法。
- [Benchmark工具](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/feature_cards/benchmark.md)：新增预置大模型训练推理 Benchmark 工具，支撑用户实现快捷部署。
- [长序列训练](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/feature_cards/Long_Sequence_Training.md)：新增支持多种长序列并行，序列长度支持至10M。
- [断点续训优化](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/resume_training.html#%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD)：断点续训场景下，优化权重和全局一致性文件保存流程，减少续训权重的校验完整性过程，加速恢复时间。
- [流水线并行优化](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/pipeline_parallel.html#interleaved-pipeline%E8%B0%83%E5%BA%A6)：提升流水线并行的效率，减少 Bubble 的占比，采用 interleaved pipeline 调度，且做了内存优化。
- [动态shape]：新增 Llama3-8B 和 Qwen2-7B 模型支持监督微调数据的输入长度动态变化。

### 新模型

以下为新支持模型：

| 模型                                                                                                           | 规格                                                                                                                 |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [Llama3.1](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/llama3_1/llama3_1.md)                | Llama3.1-8B (微调、推理)、Llama3.1-70B (微调、推理)                                                                           |
| [GLM4](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm4.md)                         | GLM4-9B (微调、推理)                                                                                                    |
| [CogVLM2_Video](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/cogvlm2_video.md)       | CogVLM2-Video-Chat-13B (微调、推理)                                                                                     |
| [CogVLM2_Image](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/cogvlm2_image.md)       | CogVLM2-Llama3-Chat-19B (推理)                                                                                       |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md)                   | Qwen1.5-0.5B (推理)、Qwen1.5-1.8B (推理)、Qwen1.5-4B (推理)、Qwen1.5-32B (推理)                                               |
| [Qwen2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen2/qwen2.md)                         | Qwen2-0.5B (微调、推理)、Qwen2-1.5B (微调、推理)、Qwen2-7B (微调、推理)、Qwen2-57B-A14B (推理)、Qwen2-57B (预训练、微调、推理)、Qwen2-72B (微调、推理) |
| [DeepSeek Coder1.5](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/deepseek1_5/deepseek1_5.md) | DeepSeek-Coder-7B-V1.5 (微调、推理)                                                                                     |
| [DeepSeekV2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/deepseek2/deepseek2.md)            | DeepSeek-V2 (预训练、微调、推理)                                                                                            |
| [Whisper](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/whisper.md)                   | Whisper-Large-V3 (微调)                                                                                              |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的 bugfix ，在此仅列举部分修复内容：

- [!3674](https://gitee.com/mindspore/mindformers/pulls/3674)：修复 Internlm2 模型解码不符合预期的问题。
- [!4401](https://gitee.com/mindspore/mindformers/pulls/4401)：修复 Baichuan2-13B 模型 MindIE 推理精度问题。

### 贡献者

感谢以下人员做出的贡献：

Chong Li，chenyijie，heqinglin，huangshengshuai，lilei，lizhihao，lizheng，moran，paolo poggi，wangshaocong，wutiancheng，xiaoshihan，yangminghai，yangzhenzhang，zhanzhan，zhaozhengquan，ZhouJingfeng，zhouyaqiang，包淦超，常少中，陈心锐，陈昱坤，陈志坚，程鹏，楚浩田，戴仁杰，冯浩，冯明昊，冯汛，耿辰华，郭儒辰，古雅诗，贺冬冬，何泽泉，胡思超，胡映彤，宦晓玲，黄磊，黄新元，黄勇，黄子灵，金仁操，孔德硕，孔紫怡，寇凯睿，蓝翔，李俊标，李洋，李文，李永文，李子垠，林鑫，林盈来，刘晨晖，刘奇，刘烙彬，刘力力，刘思铭，吕凯盟，倪钰鑫，牛君豪，邱杨，任峪瑾，赛尧，孙宇轩，唐德志，谭纬城，王浩然，汪家傲，王嘉霖，王廖辉，王双玲，魏琢艺，吴治锋，吴致远，吴昊天，杨星宇，杨犇，杨承翰，杨璇，易阳，尤日帆，俞涵，张浩，张泓铨，张吉昊，张俊杰，张敏利，张森镇，张伟，张一飞，张奕晖，张雨强，赵奕舜，周洪叶，周小琪，朱亿超，邹文祥

欢迎以任何形式对项目提供贡献！