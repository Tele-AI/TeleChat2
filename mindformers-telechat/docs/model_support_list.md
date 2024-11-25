# 模型支持列表

## NLP

### masked_language_modeling

|        模型 <br> model        | 模型规格<br>type      | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                               配置<br>config                               |
|:---------------------------:|-------------------|:----------------:|:----------------:|:---------------:|:------------------------------------------------------------------------:|
| [bert](model_cards/bert.md) | bert_base_uncased |       wiki       |        -         |        -        | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/bert) |

### [text_classification](task_cards/text_classification.md)

|                  模型 <br> model                   | 模型规格<br/>type                                              | 数据集 <br> dataset |     评估指标 <br> metric     | 评估得分 <br> score |                                配置<br>config                                |
|:------------------------------------------------:|------------------------------------------------------------|:----------------:|:------------------------:|:---------------:|:--------------------------------------------------------------------------:|
| [txtcls_bert](task_cards/text_classification.md) | txtcls_bert_base_uncased<br/>txtcls_bert_base_uncased_mnli |  Mnli <br> Mnli  | Entity F1 <br> Entity F1 | - <br>   84.80% | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/txtcls) |

### [token_classification](task_cards/token_classification.md)

|                   模型 <br> model                   | 模型规格<br/>type                                                 |   数据集 <br> dataset   |     评估指标 <br> metric     | 评估得分 <br> score  |                                配置<br>config                                |
|:-------------------------------------------------:|---------------------------------------------------------------|:--------------------:|:------------------------:|:----------------:|:--------------------------------------------------------------------------:|
| [tokcls_bert](task_cards/token_classification.md) | tokcls_bert_base_chinese<br/>tokcls_bert_base_chinese_cluener | CLUENER <br> CLUENER | Entity F1 <br> Entity F1 | - <br>    0.7905 | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/tokcls) |

### [question_answering](task_cards/question_answering.md)

|                模型 <br> model                | 模型规格<br/>type                                         |      数据集 <br> dataset      |   评估指标 <br> metric   |   评估得分 <br> score    |                              配置<br>config                              |
|:-------------------------------------------:|-------------------------------------------------------|:--------------------------:|:--------------------:|:--------------------:|:----------------------------------------------------------------------:|
| [qa_bert](task_cards/question_answering.md) | qa_bert_base_uncased<br/>qa_bert_base_chinese_uncased | SQuAD v1.1 <br> SQuAD v1.1 | EM / F1 <br> EM / F1 | 80.74 / 88.33 <br> - | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/qa) |

### translation

|      模型 <br> model      | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                              配置<br>config                              |
|:-----------------------:|---------------|:----------------:|:----------------:|:---------------:|:----------------------------------------------------------------------:|
| [t5](model_cards/t5.md) | t5_small      |      WMT16       |        -         |        -        | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/t5) |

### [text_generation](task_cards/text_generation.md)

|                      模型 <br> model                      |                                     模型规格<br/>type                                      | 数据集 <br> dataset |               评估指标 <br> metric               |                            评估得分 <br> score                            |                                  配置<br>config                                  |
|:-------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:----------------:|:--------------------------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------------------:|
|             [llama2](model_cards/llama2.md)             | llama2_7b <br/> llama2_13b <br/> llama2_7b_lora <br/> llama2_13b_lora <br/> llama2_70b |      alpaca      |                PPL / EM / F1                 | 6.58 / 39.6 / 60.5 <br/> 6.14 / 27.91 / 44.23 <br/> - <br/> - <br/> - |   [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/llama2)   |
|         [llama3](../research/llama3/llama3.md)          |                               llama3_8b <br/> llama3_70b                               |      alpaca      |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/llama3)   |
|               [glm2](model_cards/glm2.md)               |                               glm2_6b <br/> glm2_6b_lora                               |      ADGEN       | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - |     7.47 / 30.78 / 7.07 / 24.77 <br> 7.23 / 31.06 / 7.18 / 24.23      |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm2)    |
|               [glm3](model_cards/glm3.md)               |                                        glm3_6b                                         |      ADGEN       |                      -                       |                                   -                                   |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm3)    |
|               [gpt2](model_cards/gpt2.md)               |                            gpt2_small <br/> gpt2_13b <br/>                             |    wikitext-2    |                      -                       |                                   -                                   |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/gpt2)    |
|          [codellama](model_cards/codellama.md)          |                                     codellama_34b                                      |    CodeAlpaca    |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/codellama)  |
|     [baichuan2](../research/baichuan2/baichuan2.md)     |    baichuan2_7b <br/> baichuan2_13b <br/>baichuan2_7b_lora <br/> baichuan2_13b_lora    |      belle       |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan2) |
|   [deepseek coder](../research/deepseek/deepseek.md)    |                                      deepseek_33b                                      |    CodeAlpaca    |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/deepseek)  |
|         [glm32k](../research/glm32k/glm32k.md)          |                                      glm3_6b_32k                                       |    LongBench     |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/glm32k)   |
|            [Qwen](../research/qwen/qwen.md)             |                                 qwen_7b <br/> qwen_14b                                 |      alpaca      |                    C-Eval                    |                            63.3 <br/>72.13                            |   [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen)    |
|        [Qwen1.5](../research/qwen1_5/qwen1_5.md)        |                     qwen1_5_7b <br/> qwen1_5_14b <br/> qwen1_5_72b                     |      alpaca      |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen1_5)  |
|      [internlm](../research/internlm/internlm.md)       |                             internlm_7b <br/> internlm_20b                             |      alpaca      |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/internlm)  |
|     [internlm2](../research/internlm2/internlm2.md)     |                            internlm2_7b <br/> internlm2_20b                            |      alpaca      |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/internlm2) |
|        [mixtral](../research/mixtral/mixtral.md)        |                                      mixtral_8x7b                                      |    wikitext-2    |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/mixtral)  |
|               [yi](../research/yi/yi.md)                |                                   yi_6b <br/> yi_34b                                   |      alpaca      |                      -                       |                                   -                                   |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/yi)     |

## CV

### masked_image_modeling

|       模型 <br> model       | 模型规格<br/>type    | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                配置<br>config                                                 |
|:-------------------------:|------------------|:----------------:|:----------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------:|
| [mae](model_cards/mae.md) | mae_vit_base_p16 |   ImageNet-1k    |        -         |        -        | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/mae/run_mae_vit_base_p16_224_800ep.yaml) |

### [image_classification](task_cards/image_classification.md)

|        模型 <br> model        | 模型规格<br/>type  | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                配置<br>config                                                |
|:---------------------------:|----------------|:----------------:|:----------------:|:---------------:|:----------------------------------------------------------------------------------------------------------:|
|  [vit](model_cards/vit.md)  | vit_base_p16   |   ImageNet-1k    |     Accuracy     |     83.71%      |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/vit/run_vit_base_p16_224_100ep.yaml)   |
| [swin](model_cards/swin.md) | swin_base_p4w7 |   ImageNet-1k    |     Accuracy     |     83.44%      | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/swin/run_swin_base_p4w7_224_100ep.yaml) |

## Multi-Modal

### [zero_shot_image_classification](task_cards/zero_shot_image_classification.md) (by [contrastive_language_image_pretrain](task_cards/contrastive_language_image_pretrain.md))

|                  模型 <br> model                  | 模型规格<br/>type                                                            |                  数据集 <br> dataset                  |                      评估指标 <br> metric                       |              评估得分 <br> score               |                                                         配置<br>config                                                          |
|:-----------------------------------------------:|--------------------------------------------------------------------------|:--------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|
|           [clip](model_cards/clip.md)           | clip_vit_b_32<br/>clip_vit_b_16 <br/>clip_vit_l_14<br/>clip_vit_l_14@336 | Cifar100 <br> Cifar100 <br> Cifar100 <br> Cifar100 | Accuracy   <br>  Accuracy   <br>  Accuracy   <br>  Accuracy | 57.24% <br> 61.41% <br> 69.67% <br> 68.19% |       [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml)       |
| [visualglm](../research/visualglm/visualglm.md) | visualglm                                                                |                    fewshot-data                    |                              -                              |                     -                      | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/visualglm/run_visualglm_6b_image_to_text_generation.yaml) |

### image_to_text_generation

|             模型 <br> model              | 模型规格<br/>type    |   数据集 <br> dataset    | 评估指标 <br> metric | 评估得分 <br> score |                                                配置<br>config                                                |
|:--------------------------------------:|------------------|:---------------------:|:----------------:|:---------------:|:----------------------------------------------------------------------------------------------------------:|
| [QwenVL](../research/qwenvl/qwenvl.md) | qwenvl_9.6b_bf16 | LLaVa-150k detail_23k |        -         |        -        | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwenvl/finetune_qwenvl_9.6b_bf16.yaml) |

## LLM大模型能力支持一览

|     模型  \  特性     | 低参微调 |      边训边评      | Flash Attention | 并行推理  |  流式推理   |  Chat   |  多轮对话   |
|:-----------------:|:----:|:--------------:|:---------------:|:-----:|:-------:|:-------:|:-------:|
| Llama2-7B/13B/70B | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
|   Llama3-8B/70B   |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   CodeLlama-34B   | Lora |   HumanEval    |     &check;     | dp/mp | &check; |    -    |    -    |
|      GLM2-6B      | Lora | PPL/Bleu/Rouge |     &check;     | dp/mp | &check; | &check; | &check; |
|      GLM3-6B      |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|    GLM3-6B-32k    |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   GPT2-128m/13B   | Lora |      PPL       |     &check;     | dp/mp | &check; |    -    |    -    |
| BaiChuan2-7B/13B  | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
|    Qwen-7B/14B    | Lora |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|    QwenVL-9.6B    |  -   |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
|  Qwen-7B/14B/72B  |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|  InternLM-7B/20B  | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
| InternLM2-7B/20B  |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|     Yi-6B/34B     | Lora |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   Mixtral-8x7B    | Lora |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
|   DeepSeek-33B    | Lora |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
