# Llm_boost

## 功能描述

llm_boost为大模型推理加速模块, 支持对接第三方推理框架进行推理

## 支持模型

|   模型    |     硬件      | 推理  | 后端  |
| :-------: | :-----------: | :---: | :---: |
| Llama2-7b | Atlas 800T A2 | 单卡  |  ATB  |
| Qwen2-7b  | Atlas 800T A2 | 单卡  |  ATB  |

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

1. 安装CANN

- 详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)
- 安装顺序：先安装toolkit 再安装kernel

1.1 安装toolkit

- 下载

| cpu     | 包名（其中`${version}`为实际版本）               |
| ------- | ------------------------------------------------ |
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_${version}_linux-x86_64.run  |

- 安装

  ```bash
  # 安装toolkit
  chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
  ./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

1.2 安装kernel

- 下载

| 包名                                       |
| ------------------------------------------ |
| Ascend-cann-kernels-*_${version}_linux.run |

- 根据芯片型号选择对应的安装包

- 安装

  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels-*_${version}_linux.run --install
  ```

1.3 安装加速库

- 下载加速库

  | 包名（其中`${version}`为实际版本）            |
  | --------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |
  | Ascend-cann-nnal_${version}_linux-x86_64.run  |
  | ...                                           |

- 将文件放置在\${working_dir}路径下

- 安装

    ```bash
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run --install --install-path=${working_dir}
    source ${working_dir}/nnal/atb/set_env.sh
    ```

1.3 安装atb_models

  ```bash
  mkdir atb-models
  cd atb-models
  tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi0.tar.gz
  source set_env.sh
  ```

#### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，`vocab.json`和`merges.txt`文件也在链接中下载。

词表下载链接：[vocab.json](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/vocab.json)和[merges.txt](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/merges.txt)

| 模型名称          |                                     Base权重（建议训练和微调使用）                                     |                  Instruct权重（建议推理使用）                   |
| :---------------- | :----------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------: |
| llama2-7b         | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt) |     [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)     |
| qwen2-7b-Instruct |                         [Link](https://huggingface.co/Qwen/Qwen2-7B/tree/main)                         | [Link](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
以Llama2-7b为例。
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

```shell
运行mindformers/mindformers/llm_boost/convert_weight.py转换脚本，将ckpt中qkv权重进行合并
python mindformers/mindformers/llm_boost/convert_weight.py  --pre_ckpt_path PRE_CKPT_DIR --mindspore_ckpt_path OUTPUT_CKPT_DIR -qkv_concat True

# 参数说明
pre_ckpt_path:       转换前的权重路径
mindspore_ckpt_path: 转换后的权重路径
```

### 单卡推理

推理流程与原模型推理流程一致，只需修改配置文件

以`Llama2-7b`单卡推理为例。

```shell
# model config
model:
  model_config:
    type: LlmBoostConfig
    llm_backend: BuildIn  # llm backend
    boost_model_name: Llama # model name
    batch_size: 1 # add for increase predict
    seq_length: 4096
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    vocab_size: 32000
    rms_norm_eps: 1.0e-5
    bos_token_id: 1
    eos_token_id: 2
    pad_token_id: 0
    ignore_token_id: -100
    compute_dtype: "float16"
    rotary_dtype: "float16"
    use_past: True
    scaling_factor: 1.0 # The scale factor of seq length
    extend_method: "None" # support "None", "PI", "NTK"
    block_size: 16
    num_blocks: 1024
    is_dynamic: True
    offset: 0
    repetition_penalty: 1
    max_decode_length: 512
    top_k: 3
    top_p: 1
    do_sample: False
    need_nz: False
    communication_backend: lccl
  arch:
    type: LlmBoostForCausalLM
```
