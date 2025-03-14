# Harness评测

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindFormers进行适配后，支持加载MindFormers模型进行评测。

## 安装

```shell
pip install lm_eval==0.4.4
```

## 使用方式

### 查看数据集评测任务

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

### 启动单卡评测脚本

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
```

### 启动多卡并行评测脚本

```shell
#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

bash  mindformers/scripts/msrun_launcher.sh "toolkit/benchmarks/eval_with_harness.py \
    --model mf \
    --model_args pretrained=MODEL_DIR,use_parallel=True,tp=1,dp=4 \
    --tasks TASKS \
    --batch_size 4" 4
```

可通过环境变量ASCEND_RT_VISIBLE_DEVICES设置多卡卡号。

### 评测参数

#### Harness主要参数

| 参数            | 类型  | 参数介绍                      | 是否必须 |
|---------------|-----|---------------------------|------|
| --model       | str | 须设置为mf，对应为MindFormers评估策略 | 是    |
| --model_args  | str | 模型及评估相关参数，见下方模型参数介绍       | 是    |
| --tasks       | str | 数据集名称，可传入多个数据集，逗号分割       | 是    |
| --batch_size  | int | 批处理样本数                    | 否    |
| --num_fewshot | int | Few_shot的样本数              | 否    |
| --limit       | int | 每个任务的样本数，多用于功能测试          | 否    |

#### MindFormers模型参数

| 参数           | 类型   | 参数介绍                              | 是否必须 |
|--------------|------|-----------------------------------|------|
| pretrained   | str  | 模型目录路径                            | 是    |
| use_past     | bool | 是否开启增量推理，generate_until类型的评测任务须开启 | 否    |
| device_id    | int  | 设备id                              | 否    |
| use_parallel | bool | 开启并行策略                            | 否    |
| dp           | int  | 数据并行                              | 否    |
| tp           | int  | 模型并行                              | 否    |

### 评测前准备

1. 创建模型目录MODEL_DIR；
2. 模型目录下须放置MindFormers权重、yaml配置文件、分词器文件，获取方式参考MindFormers模型README文档；
3. 配置yaml配置文件。

yaml配置参考：

```yaml
run_mode: 'predict'
model:
  model_config:
    checkpoint_name_or_path: "model.ckpt"
processor:
  tokenizer:
    vocab_file: "tokenizer.model"
```

## 评测样例

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=./llama3-8b,use_past=True" --tasks gsm8k

# 评估结果如下，其中Filter对应匹配模型输出结果的方式，Metric对应评估指标，Value对应评估分数，stderr对应分数误差。
# mf (pretrained=./llama3-8b), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5034|±  |0.0138|
# |     |       |strict-match    |     5|exact_match|↑  |0.5011|±  |0.0138|
```

## 支持特性说明

支持Harness全量评测任务，包含如下任务组及其子任务：

<details>
<summary>点击展开Harness评测任务表</summary>

| Group           | Task                                                                      |
|-----------------|---------------------------------------------------------------------------|
| aclue           | aclue_ancient_chinese_culture, aclue_ancient_literature ...               |
| aexams          | aexams_Biology, aexams_IslamicStudies ...                                 |
| agieval         | agieval_aqua_rat, agieval_math ...                                        |
| arabicmmlu      | arabicmmlu_driving_test, arabicmmlu_general_knowledge ...                 |
| bbh             | bbh_cot_fewshot_boolean_expressions, bbh_cot_fewshot_causal_judgement ... |
| belebele        | belebele_acm_Arab, belebele_afr_Latn, ...                                 |
| blimp           | blimp_adjunct_island, blimp_anaphor_gender_agreement ...                  |
| ceval-valid     | ceval-valid_accountant, ceval-valid_advanced_mathematics ...              |
| cmmlu           | cmmlu_agronomy, cmmlu_anatomy ...                                         |
| csatqa          | csatqa_gr, csatqa_li ...                                                  |
| flan            | anli_r1_flan, anli_r2_flan ...                                            |
| haerae          | haerae_general_knowledge, haerae_history ...                              |
| hendrycks_math  | hendrycks_math_algebra, hendrycks_math_counting_and_prob ...              |
| kormedmcqa      | kormedmcqa_doctor, kormedmcqa_nurse ...                                   |
| leaderboard     | leaderboard_bbh_boolean_expressions, leaderboard_bbh_causal_judgement ... |
| lingoly         | lingoly_context, lingoly_nocontext ...                                    |
| med_concepts_qa | med_concepts_qa_atc_easy, med_concepts_qa_atc_hard ...                    |
| mela            | mela_ar, mela_de ...                                                      |
| minerva_math    | minerva_math_algebra, minerva_math_counting_and_prob ...                  |
| mmlu            | mmlu_abstract_algebra, mmlu_abstract_algebra_generative ...               |
| multimedqa      | mmlu_anatomy, mmlu_clinical_knowledge ...                                 |
| openllm         | arc_challenge, hellaswag ...                                              |
| pawsx           | paws_de, paws_en ...                                                      |
| pythia          | lambada_openai, logiqa ...                                                |
| t0_eval         | anli_r1, anli_r2 ...                                                      |
| tinyBenchmarks  | tinyGSM8k, tinyHellaswag ...                                              |
| tmlu            | tmlu_AST_biology, tmlu_AST_chemistry ...                                  |
| tmmluplus       | tmmluplus_accounting, tmmluplus_administrative_law ...                    |
| wmdp            | wmdp_bio, wmdp_chem ...                                                   |
| xcopa           | xcopa_et, xcopa_ht ...                                                    |
| xnli            | xnli_ar, xnli_bg ...                                                      |
| xstorycloze     | xstorycloze_ar, xstorycloze_en ...                                        |
| xwinograd       | xwinograd_en, xwinograd_fr ...                                            |

</details>

具体评测任务见[查看数据集评测任务](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Harness_Eval.md#%E6%9F%A5%E7%9C%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AF%84%E6%B5%8B%E4%BB%BB%E5%8A%A1)。