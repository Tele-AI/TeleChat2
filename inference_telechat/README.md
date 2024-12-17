# conda创建推理和微调环境

**默认前置知识**：conda的基本使用

**环境参考：**

| Linux              | GPU         | CUDA Version |
| ------------------ | ----------- | ------------ |
| ubuntu22.04 x86_64 | A800 80GB*4 | 12.4         |

CUDA Version需要满足：CUDA Version>=cuda toolkit version>=11.7

A800 80GB*4对于仓库中所给推理示例，GPU占用量：

![image-20241217141142481](https://ycy-lenovo-typora.oss-cn-beijing.aliyuncs.com/typora_imgs/image-20241217141142481.png)

A800 80GB*4对于仓库中所给参考数据，**lora微调**的GPU占用量：

![image-20241217111035432](https://ycy-lenovo-typora.oss-cn-beijing.aliyuncs.com/typora_imgs/image-20241217111035432.png)

```
conda create -n tele python=3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.45.0 
```

直接“pip install deepspeed==0.9.3”会报错”AssertionError: CUDA_HOME does not exist, unable to compile CUDA op(s)“，通常在安装CUDA Toolkit后，需要手动设置 `CUDA_HOME` 环境变量。

先安装CUDA Toolkit:`nvcc` 是 CUDA Toolkit 中的编译器，如果安装了 CUDA Toolkit，应该可以通过命令行`nvcc --version`调用 `nvcc`。

同时后面FlashAttention的包flash-attn要求 CUDA 版本为 **11.7** 或更高版本

不要`sudo apt install nvidia-cuda-toolkit`安装cuda toolkit，这里的默认版本是11.5的，cuda toolkit用了11.8

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

然后

```
pip install deepspeed==0.9.3
pip install flash-attn==2.0.0.post1
pip install sentencepiece
pip install accelerate>=0.34.2
```

如果出现报错:A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

```
pip install "numpy<2"
```

如果出现NameError:name'Extension' is not defined,Did you mean:'Exception'?

检查一下是否安装了jinja2这个包，如果没安装，`jinja2.ext` 模块中的 `Extension` 类没有正确导入，就会报这个错

```
pip install jinja2
```

然后这样就可以运行telechat_infer_demo.py

##  微调 

在以上推理环境中，继续

```
pip install datasets
pip install peft
```

将TeleChat2/deepspeed/train_scripts/[sft_single_node.sh](https://github.com/Tele-AI/TeleChat2/blob/main/deepspeed/train_scripts/sft_single_node.sh)

复制到TeleChat2/deepspeed/ 目录下，即和train.py同级

脚本中存在 **回车符 (Carriage Return, `\r`)**

使用sed命令删除回车符：

```
sed -i 's/\r//g' sft_single_node.sh
```

根据所在系统GPU数量、模型存放路径，修改相应参数

![image-20241217142006825](https://ycy-lenovo-typora.oss-cn-beijing.aliyuncs.com/typora_imgs/image-20241217142006825.png)

bash sft_single_node.sh 即可