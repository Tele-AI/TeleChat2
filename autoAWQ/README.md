# TeleChat Autoawq 推理使用方式

## 下载autoawq
```
pip install autoawq==x.x.x
pip install accelerate==0.34.2
pip install flash-attn==2.6.3
```

## autoawq 添加telechat

### 将telechat model文件放入
pip show autoawq 找到autoawq对应位置并进入
```bash
cd ./awq/models/
```
将此路径下的 telechat.py和bsae.py 文件放入以上路径


```bash
cd ./awq/quantize/
```
将此路径下的 quantize.py 替换原 quantize.py 文件

### 修改init文件
修改同路径下的__init__.py
```python
from .minicpm3 import MiniCPM3AWQForCausalLM
from .qwen2vl import Qwen2VLAWQForCausalLM
from .telechat2 import Telechat2AWQForCausalLM # 添加telechat2
```

### 修改auto文件
修改同路径下的auto.py
```python
    "minicpm3": MiniCPM3AWQForCausalLM,
    "qwen2_vl": Qwen2VLAWQForCausalLM,
    'telechat': Telechat2AWQForCausalLM, # 添加telechat2
```


# 启动
使用autoawq 量化telechat2模型
```bash
python telechat_quant.py
```