# unified_safetensors.py 使用教程

## 脚本说明

该脚本是用来合并训练之后保存的权重（支持去除冗余的权重），且保存后的格式为`safetensors`

## 使用说明

```python
python unified_safetensors.py \
  --mindspore_ckpt_dir /path/checkpoint \
  --src_strategy_dirs /path/src_strategy_dirs \
  --output_dir /path/src_strategy_dirs \
  --file_suffix "1_1" \
  --format "ckpt" \
  --has_redundancy False \

参数说明：

- mindspore_ckpt_dir: 权重路径，需指定rank_*上级目录
- src_strategy_dirs：分布式策略文件路径
- output_dir：保存合并后的权重路径
- file_suffix：权重保存后的后缀，如"1_1"
- format：源权重格式，"ckpt" 或者是 "safetensors"
- has_redundancy：是否是去除冗余后的权重
```

