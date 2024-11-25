# 边训练边评估

大模型的训练效果需要评测任务来作为衡量标准，而当前大模型的训练耗时长，等到训练整体结束后再进行评测任务的时间与算力成本过高

本功能提供了一套在训练过程中进行评估的流程方法，以动态观察模型的评估结果，具有以下特性：

1. 能够在训练过程中执行评估流程，并在日志中打印相关的评估结果信息；
2. 具有功能开关，根据开关状态决定是否启用边训练边评估功能；
3. 具备调用间隔控制项，根据该控制项决定调用边训练边评估功能的时间间隔；
4. 支持多种模型，只要模型能够调用Model.eval()完成评估，都可以复用该项功能；无法通过Model.eval()完成评估的模型则需要额外编码适配，暂不支持；

## [边训练边评估支持度表](../model_support_list.md#边训练边评估支持度表)

通过标题链接可查看边训练边评估特性的支持度表格

说明：边训练边评估功能需模型已支持评估，并且该评估指标可以通过Model.eval()完成

## 使用说明

### run_mindformer启用边训练边评估

**描述**：通过run_mindformers脚本参数，启用边训练边评估功能

**使用方式**：`--do_eval`开关启用边训练边评估功能，`--eval_dataset_dir`指定评估数据集

```shell
python run_mindformer.py \
--config configs/gpt2/pretrain_gpt2_small_fp16.yaml \
--run_mode train \
--train_dataset_dir /your_path/wikitext-2-train-mindrecord \
--eval_dataset_dir /your_path/wikitext-2-eval-mindrecord \
--do_eval True
```

### trainer启用边训练边评估

**描述**：通过Trainer高阶接口入参，启用边训练边评估功能

**使用方式**：执行以下python脚本，其中数据集路径替换为实际路径

```python
def test_trainer_do_eval():
    import mindspore as ms
    from mindformers.trainer import Trainer

    ms.set_context(mode=0) # 设定为图模式加速
    # 初始化预训练任务
    trainer = Trainer(task='text_generation', model='gpt2',
                      train_dataset="/your_path/wikitext-2-train-mindrecord",
                      eval_dataset="/your_path/wikitext-2-eval-mindrecord")
    # 开启训练，并打开do_eval开关
    trainer.train(do_eval=True)

if __name__ == "__main__":
    test_trainer_do_eval()
```

### 配置评估间隔时间

**描述**：更改评估间隔时间，以控制执行评估的频率

mindformers通过 `eval_step_interval` 和 `eval_epoch_interval` 两项配置参数来控制边训练边评估的执行间隔，参数含义如下：

- **eval_step_interval**: 评估step间隔, 默认为100，表示每100个step间隔执行一次评估；配置为大于0的数表示每隔所配置的step数后执行一次评估，配置为小于0的数则表示禁用step评估；注意：在数据下沉模式下，step间隔值建议配置为sink size的倍数
- **eval_epoch_interval**: 评估epoch间隔, 默认为-1，表示禁用epoch结束时的评估；配置为大于0的数表示每隔所配置的epoch数后执行一次评估，配置为小于0的数则表示禁用epoch评估；注意：数据下沉模式下，epoch所包含的step数将从数据集大小变为sink size的大小，将在 `sink_size * eval_epoch_interval` 个step后执行一次评估

**使用方式1**：脚本启动时，可以通过更改配置项，在 `configs/gpt2/run_gpt2.yaml` 文件中新增 `eval_step_interval` 或 `eval_epoch_interval` 项修改为期望的执行间隔数值，执行 `run_mindformer.py` 启用边训练边评估中的启动脚本

```yaml
# 边训练边评估开关，True启用，False关闭，脚本传参的do_eval将覆盖该项配置
do_eval: False
# 配置评估step间隔
eval_step_interval: 50
# 配置评估epoch间隔
eval_epoch_interval: -1     # 默认值-1表示禁用epoch结束时的评估
```

**使用方式2**：高阶接口使用时，通过传入TrainingArguments以修改执行间隔配置数值

```python
def test_trainer_do_eval():
    import mindspore as ms
    from mindformers.trainer import Trainer, TrainingArguments

    ms.set_context(mode=0) # 设定为图模式加速

    args = TrainingArguments(
        batch_size=4,
        num_train_epochs=1,
        eval_step_interval=50,      # 通过TraingArguments修改评估间隔
        eval_epoch_interval=-1      # 默认值-1表示禁用epoch结束时的评估
    )

    # 初始化预训练任务，传入args以配置评估间隔
    trainer = Trainer(task='text_generation', model='gpt2',
                      args=args,
                      train_dataset="/your_path/wikitext-2-train-mindrecord",
                      eval_dataset="/your_path/wikitext-2-eval-mindrecord")

    # trainer实例化之后也可通过trainer.config对配置进行修改，如下
    # trainer.config.eval_step_interval = 100
    # trainer.config.eval_epoch_interval = -1

    # 开启训练，并打开do_eval开关
    trainer.train(do_eval=True)

if __name__ == "__main__":
    test_trainer_do_eval()
```

## 查看评估结果

按照上述方式启动边训练边评估后，查看训练日志，可通过 `Eval result` 关键字检索训练日志以查看训练中评估的结果，如下图：

![gpt2-runmindormer](https://foruda.gitee.com/images/1686903702963042587/d2c01f36_7579591.png "gpt2-ppl.png")

> 注: 评估需要对评估数据集进行全量评估，通常较为耗时，建议将设置较长的评估间隔以减少对训练性能的影响
