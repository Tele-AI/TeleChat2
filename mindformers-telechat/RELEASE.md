# MindSpore Transformers Release Notes

## MindSpore Transformers 1.3.0 Release Notes

Below is the changelog for MindSpore Transformers (referred to as MindFormers) version 1.3.0, highlighting key new features and bug fixes compared to version 1.2.0.

### New Features

- [Installation Verification](https://www.mindspore.cn/mindformers/docs/en/dev/mindformers/mindformers.run_check.html): Added a convenient API to check whether MindFormers has been successfully installed.
- [Log Optimization]: Optimized MindFormers logs, providing more comprehensive information to improve accuracy in issue identification and monitoring of training status.
- [LLM Chat API](https://www.mindspore.cn/mindformers/docs/en/dev/generation/mindformers.generation.GenerationMixin.html#mindformers.generation.GenerationMixin.chat): Provides a text generation inference interface for large language model-based conversations.
- [Quantization Inference](https://www.mindspore.cn/mindformers/docs/en/dev/usage/quantization.html): Integrated with MindSpore Golden Stick toolset, offering a unified quantization inference process.
- [BIN Format Dataset](https://www.mindspore.cn/mindformers/docs/en/dev/function/dataset.html#bin-format-dataset): Added the ability to work with BIN format datasets, including how to make BIN format datasets and use BIN format datasets in tasks.
- [Online Dataset](https://www.mindspore.cn/mindformers/docs/en/dev/function/dataset.html#online-dataset): Supports loading online datasets during training, eliminating the need for local offline preprocessing.
- [Evaluation](https://www.mindspore.cn/mindformers/docs/en/dev/usage/evaluation.html): Based on the Harness evaluation framework, MindFormers models can be loaded for evaluation with custom prompts and evaluation metrics, including loglikelihood, generate_until, and loglikelihood_rolling tasks.
  Based on the VLMEvalKit evaluation framework, the system also supports evaluation of multimodal large models with custom prompts and metrics, including MME, MMEBench, and COCO caption methods for image-text understanding tasks.
- [Benchmark Tools]: Added preset large-model training and inference benchmarking tools to enable users to deploy quickly.
- [Long-Sequence Training]: Added support for multiple long-sequence parallelism, with sequence lengths up to 10M.
- [Checkpoint Resumption Optimization](https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html#resumable-training): Improved the saving process for weights and global consistency files during checkpoint resumption, reducing the time needed to verify weight integrity and speeding up recovery.
- [Pipeline Parallelism Optimization]: improved the efficiency of pipeline parallelism and reduce the proportion of bubbles, adopted interleaved pipeline scheduling with memory optimization.
- [Dynamic Shape]: Added support for dynamic input length of supervised fine-tuning data for the Llama3_8B and Qwen2_7B models.

### New Models

The following new models are now supported:

| Model               | Specifications                                                                                                                                                                                             |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Llama3.1]          | Llama3.1-8B (finetune, inference), Llama3.1-70B (finetune, inference)                                                                                                                                      |
| [GLM4]              | Glm4-9B (finetune, inference)                                                                                                                                                                              |
| [CogVLM2_Video]     | CogVLM2-Video-Chat-13B (finetune, inference)                                                                                                                                                               |
| [CogVLM2_Image]     | CogVLM2-Llama3-Chat-19B (inference)                                                                                                                                                                        |
| [Qwen1.5]           | Qwen1.5-0.5B (inference), Qwen1.5-1.8B (inference), Qwen1.5-4B (inference), Qwen1.5-32B (inference)                                                                                                        |
| [Qwen2]             | Qwen2-0.5B (finetune, inference), Qwen2-1.5B (finetune, inference), Qwen2-7B (finetune, inference), Qwen2-57B-A14B (inference), Qwen2-57B (pretrain, finetune, inference), Qwen2-72B (finetune, inference) |
| [DeepSeek Coder1.5] | DeepSeek-Coder-7B-V1.5 (finetune, inference)                                                                                                                                                               |
| [DeepSeekV2]        | DeepSeek-V2 (pretrain, finetune, inference)                                                                                                                                                                |
| [Whisper]           | Whisper-Large-V3 (finetune)                                                                                                                                                                                |

### Bugfix

During this release cycle, we addressed numerous bugs across models, functionalities, usability, and documentation.
Here are some notable fixes:

- [!3674]: Fixed an issue with the Internlm2 model not decoding as expected.
- [!4401]: Fixed the issue with inference accuracy for the Baichuan2_13B model in MindIE.

### Contributors

Thanks to the following individuals for their contributions:

Chong Li, chenyijie, heqinglin, huangshengshuai, lilei, lizhihao, lizheng, moran, paolo poggi, wangshaocong, wutiancheng, xiaoshihan, yangminghai, yangzhenzhang, zhanzhan, zhaozhengquan, ZhouJingfeng, zhouyaqiang, 包淦超, 常少中, 陈心锐, 陈昱坤, 陈志坚, 程鹏, 楚浩田, 戴仁杰, 冯浩, 冯明昊, 冯汛, 耿辰华, 郭儒辰, 古雅诗, 贺冬冬, 何泽泉, 胡思超, 胡映彤, 宦晓玲, 黄磊, 黄新元, 黄勇, 黄子灵, 金仁操, 孔德硕, 孔紫怡, 寇凯睿, 蓝翔, 李俊标, 李洋, 李文, 李永文, 李子垠, 林鑫, 林盈来, 刘晨晖, 刘奇, 刘烙彬, 刘力力, 刘思铭, 吕凯盟, 倪钰鑫, 牛君豪, 邱杨, 任峪瑾, 赛尧, 孙宇轩, 唐德志, 谭纬城, 王浩然, 汪家傲, 王嘉霖, 王廖辉, 王双玲, 魏琢艺, 吴治锋, 吴致远, 吴昊天, 杨星宇, 杨犇, 杨承翰, 杨璇, 易阳, 尤日帆, 俞涵, 张浩, 张泓铨, 张吉昊, 张俊杰, 张敏利, 张森镇, 张伟, 张一飞, 张奕晖, 张雨强, 赵奕舜, 周洪叶, 周小琪, 朱亿超, 邹文祥

Contributions to the project in any form are welcome!