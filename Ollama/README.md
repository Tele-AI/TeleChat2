# Ollama
本项目主要将telechat适配Ollama框架

分支说明：
build_docker分支仅修改了必要的文件，便于之后提交到telechat的github仓库。同时增加了go语言的离线依赖库vendor文件夹，便于进行后续的编译开发。

## 支持列表

项目适配了telechat2系列模型(7b、35b、115b), 在下列平台上经过验证可以编译成功。

|  计算能力  | 架构名称 | 验证显卡  |
|  ----   | ----  | ----  |
| sm80  | Ampere | A100 |
| sm86  | Ampere | A10 |
| sm89  | Ada Lovelace	 | 4090 |

## 项目编译
镜像环境中配置了go环境的依赖，可直接进行编译，也可直接通过go.mod下载go相应的依赖

```bash
go mod tidy
```

在编译的时候可以根据显卡制定编译的计算能力，从而加快项目的编译速度。

```bash
vim make\Makefile.cuda_v12
```
默认编译80;86;89这几款主流架构的显卡
```yaml
CUDA_ARCHITECTURES?=80;86;89
```
执行make指令开始编译，编译成功后可以看到Ollama
```bash
make -j 5
```

## 项目执行
模型转换需要在开发镜像中进行，其余操作只需部署镜像即可

### 模型转换
```bash
cd /opt/Ollama
python3 llama/convert/convert_hf_to_gguf.py ${HF_MODEL_PATH}
```

转换成功显示：INFO:hf-to-gguf:Model successfully.

###  启动ollama服务
通过常用环境变量配置Ollama的服务
指定gpu
```bash
export CUDA_VISIBLE_DEVICES=7
```

设置模型存活时间
默认5m就是五分钟 （如：纯数字如 300 代表 300 秒，0 代表处理请求响应后立即卸载模型，任何负数则表示一直存活
```bash
export OLLAMA_KEEP_ALIVE=-1
```
设置模型并发限制
```bash
export OLLAMA_NUM_PARALLEL=8
```
设置模型存储位置
ollama会将gguf文件缓存到默认目录~/.ollama，如果需要指定到容器外
```bash
export OLLAMA_MODELS=/mnt/dir
```
启动Ollama
设置完成后可以启动Ollama服务
```bash
ollama serve
```

### 创建模型
创建模型文件
```bash
vim 7B.Modelfile
```

将转换成功的模型地址写入
```bash
FROM /home/hf-model/telechat2_7b/Telechat2_7B-7.6B-F16.gguf
```

执行ollama create image:tag 命令 telechat2是模型名，如果不指定的话会默认给一个latest的tag
```bash
ollama create telechat2 -f 7B.Modelfile 
```

成功之后执行./ollama list可以看到模型
```bash
ollama list
```

### 运行模型

命令行对话
```bash
ollama run telechat2:latest
```

可以粘贴下面的样例，查看输出结果

```text
<_system>你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。<_user>北京有哪些经典美食<_bot>
```

http请求

默认需要请求一次，然后模型才会被加载，加载后存活时间可以通过环境变量或请求参数设定

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{  "model": "telechat2:latest",  "messages": [ {"role": "user", "content": "Why is the sky blue?"}  ],  "temperature": 0.7,  "top_p": 0.8, "max_tokens": 30, "stream": false}'
```

ollama提供自定义的原生接口，也有兼容openai格式的接口，具体可以参考官方文档https://ollama.readthedocs.io/api/ 

