# Qwen-Agent
[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) 是一个基于 Qwen 的指令跟随、工具使用、计划和记忆能力来开发 LLM 应用程序的框架。它还附带了一些示例应用程序，例如浏览器助手、代码解释器和自定义助手。
现已经可以通过vllm启动telechat服务 使用Qwen-Agent

## 安装
```
git clone https://github.com/QwenLM/Qwen-Agent.git
cd Qwen-Agent
pip install -e ./
```

## 开发您自己的智能体
```
import json
import os

import json5
import urllib.parse
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

llm_cfg = {
    # Use your own model service compatible with OpenAI API:
    # 'model': 'TeleChat2/TeleChat2-7B',
    # 'model_server': 'http://localhost:8000/v1',  # api_base
    # 'api_key': 'EMPTY',

    # (Optional) LLM hyperparameters for generation:
    'generate_cfg': {
        'top_p': 0.8
    }
}
system = 'According to the user\'s request, you first draw a picture and then automatically run code to download the picture ' + \
          'and select an image operation from the given document to process the image'

# Add a custom tool named my_image_gen：
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI painting (image generation) service, input text description, and return the image URL drawn based on text information.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the desired image content, in English',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


tools = ['my_image_gen', 'code_interpreter']  # code_interpreter is a built-in tool in Qwen-Agent
bot = Assistant(llm=llm_cfg,
                system_message=system,
                function_list=tools,
                files=[os.path.abspath('doc.pdf')])

messages = []
while True:
    query = input('user question: ')
    messages.append({'role': 'user', 'content': query})
    response = []
    for response in bot.run(messages=messages):
        print('bot response:', response)
    messages.extend(response)
```