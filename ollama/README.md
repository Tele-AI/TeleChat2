# Ollama
项目将telechat2系列模型(7b、35b、115b)适配Ollama框架。

## 项目配置

### 下载代码
拉取Ollama项目
```bash
git clone https://github.com/ollama/ollama.git && cd ollama
```

切换代码版本
```bash
git checkout -b add_telechat 297ada6c8
```

将拉取的ollama项目加入安全路径,避免后续编译报错
```bash
git config --global --add safe.direcotry ~/ollama
```

### 替换核心文件
需要对官方ollama项目增加一个文件夹，替换一个文件即可完成对TeleChat2模型对Ollama的适配

1. 进入Telechat2项目的ollama路径
```bash
cd ollama
```

2. 将convert文件夹拷贝到ollama项目的llama文件夹下
```bash
cp -r convert ~/ollama/llama
```

3. 替换llama.cpp文件
```bash
cp llama.cpp ~/ollama/llama
```

## 项目编译
可通过go.mod下载go相应的依赖

```bash
go mod tidy
```

也准备了go1.23.4离线依赖包，可以直接下载，解压到ollama下vendor文件夹
[go依赖下载链接](https://pan.baidu.com/s/1ptbWpfv3ka5w6YbkNblauQ)
```bash
unzip -d ~/ollama vendor.zip
```

编译的时可以根据显卡制定编译的计算能力，选择本机显卡加快项目的编译速度。
```bash
vim make/Makefile.cuda_v12
```
默认编译编译所有架构的显卡，如下所示可以指定编译80 86 89计算能力的显卡。
```yaml
CUDA_ARCHITECTURES?=80;86;89
```
执行make指令开始编译，编译成功后可以看到目录下有ollama的可执行文件，可以将ollama软连接到/usr/bin下面便于使用
```bash
make -j 5
```

## 项目执行
### 模型转换
```bash
cd ollama
python3 llama/convert/convert_hf_to_gguf.py ${HF_MODEL_PATH}
```

转换成功显示：INFO:hf-to-gguf:Model successfully.

###  启动ollama服务
通过常用环境变量配置Ollama的服务
指定gpu
```bash
export CUDA_VISIBLE_DEVICES=0
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
ollama会将gguf文件缓存到默认目录~/.ollama，如果需要更改存储位置设置如下环境变量
```bash
export OLLAMA_MODELS=/mnt/dir
```
设置ollama仅可本地访问, 默认是0.0.0.0允许所有设备访问
```bash
export OLLAMA_HOST=127.0.0.1
```
启动Ollama
设置完成后可以启动Ollama服务
```bash
./ollama serve
```

### 创建模型
创建模型文件
```bash
vim 7B.Modelfile
```

将转换成功的gguf模型地址写入7B.Modelfile
```bash
FROM /home/hf-model/telechat2_7b/Telechat2_7B-7.6B-F16.gguf
```

执行ollama create image:tag 命令 telechat2是模型名，如果不指定的话会默认给一个latest的tag
```bash
./ollama create telechat2 -f 7B.Modelfile 
```

成功之后执行./ollama list可以看到创建成功的模型
```bash
./ollama list
```

### 运行模型
**命令行对话**
```bash
./ollama run telechat2:latest
```

可以粘贴下面的样例，查看输出结果
```text
<_system>你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。<_user>北京有哪些美食<_bot>
```

**http请求**

请默认需要请求一次，然后模型才会被加载，加载后存活时间可以通过环境变量或请求参数设定

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{  "model": "telechat2:latest",  "messages": [ {"role": "user", "content": "Why is the sky blue?"}  ],  "temperature": 0.7,  "top_p": 0.8, "max_tokens": 30, "stream": false}'
```

ollama提供自定义的原生接口，也有兼容openai格式的接口，具体可以参考[官方文档api接口](https://ollama.readthedocs.io/api/)

