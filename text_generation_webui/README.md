# Text-generation-webui部署流程

## 下载代码
```bash
git clone https://github.com/oobabooga/text-generation-webui

```

建议版本：v1.16

## 替换核心文件
共需要替换2个文件即可完成对TeleChat模型的适配：

进入text-generation-webui目录：
```bash
cd ./text-generation-webui
```

1. 使用当前目录下的"shared.py"文件替换text-generation-webui目录下的"text-generation-webui/modules/shared.py"

其中，"shared.py"文件的具体修改为：
修改默认对话模式为chat（35行）：
```python
'mode': 'chat',
```
修改默认system指令（59行）：
```python
'custom_system_message': '<_system>你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。',
```

2. 使用当前目录下的"chat.py"文件替换text-generation-webui目录下的"text-generation-webui/modules/chat.py"

具体修改的代码为：

设置特殊token（106行）：
```python
state["name1"] = "<_user>"
state["name2"] = "<_bot>"
state["context"] = ""
```

添加system指令（125行）：
```python
if state['custom_system_message'].strip() != '':
    messages.append({"role": "system", "content": state['custom_system_message']})
```

添加结束符（137行）：

```python
if assistant_msg:
    messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg + "<_end>"})
```


## 准备模型
模型从Hugging Face上进行下载，并且存放在：text-generation-webui/models. 

例如：
```
text-generation-webui
|—— models
│   |—— telechat2-7B
│   │   |—— config.json
|   |   |—— modeling_telechat2
│   │   |—— generation_config.json
│   │   |—— pytorch_model-00001-of-00004.bin
│   │   |—— pytorch_model-00002-of-00004.bin
│   │   |—— pytorch_model-00003-of-00004.bin
│   │   |—— pytorch_model-00004-of-00004.bin
│   │   ├── tokenization_telechat2.bin.index.json
│   │   |—— generation_utils.py
│   │   |—— pytorch_model.bin.index.json
│   │   |—— configuration_telechat2.py
```

## 启动方式
```bash 
CUDA_VISIBLE_DEVICES=0 python one_click.py --listen --listen-host=0.0.0.0 --trust-remote-code
```

运行成功后访问：http://127.0.0.1:7860/