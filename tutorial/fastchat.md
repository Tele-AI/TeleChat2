# FastChat模型部署流程

## 环境安装
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat

pip3 install -e ".[model_worker,webui]"
```

## 替换核心文件
共需要替换两个文件即可完成对TeleChat模型的适配：
```bash 
pip show fastchat
```
找到fastchat安装目录并进入：
```bash
cd ./fastchat
```
1. 使用fastchat_serving目录下的[conversation.py](../fastchat_serving/conversation.py)文件替换fastchat安装目录下的"fastchat/conversation.py"
2. 使用fastchat_serving目录下的[model_adapter.py](../fastchat_serving/model_adapter.py)文件替换fastchat安装目录下的"fastchat/model/model_adapter.py"

## 全局控制器 controller 
controller能够进行模型注册


```bash
python3 -m fastchat.serve.controller
```

下载模型到当前部署目录，模型名称分别为 "model_1", "model_2"，具体名称根据下载的模型决定。

## 启动单个模型: "model_1"
```bash
python3 -m fastchat.serve.model_worker --model-path ./model_1 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
```

## 发送问题测试模型是否被注册
python3 -m fastchat.serve.test_message --model-name model_1

如果能得到正常回复，则说明该模型已经被部署成功

## 启动单模型webui
```bash
python3 -m fastchat.serve.gradio_web_server
```
启动webui后，便可以得到端口号，其他人通过该端口便可以访问webui。例如可以访问 http://127.0.0.1:7860


# 多模型部署
首先启动控制器，然后开始启动多个模型，并且把每个模型跟控制器通过端口号进行连接。

模型1
```bash
python3 -m fastchat.serve.model_worker --model-path ./model_1 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
```

模型2

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ./model_2 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
```


## 多模型webui
```bash

python3 -m fastchat.serve.gradio_web_server_multi
```
启动后可以查看到已经注册成功的模型名称与数量，以及访问webui访问地址。

# vLLM部署

需要设置spawn分布式方式
```bash 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m fastchat.serve.vllm_worker --model-path ./model_1 --controller http://localhost:21001 --port 31002 --tensor-parallel-size 4 --worker-address http://localhost:31002
```