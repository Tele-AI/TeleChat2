# 安装依赖（测试已通过版本)
```
pip install autoawq
pip install accelerate==0.34.2
pip install flash-attn==2.6.3
```
# 修改telechat_quant.py中的路径
# 执行telechat_quant.py
执行量化步骤
# 修改load_quant_model_test.py中的路径并进行测试
测试量化模型的性能，实测1024长度下，35B-A100-40g可以从2卡资源需求变为1卡。