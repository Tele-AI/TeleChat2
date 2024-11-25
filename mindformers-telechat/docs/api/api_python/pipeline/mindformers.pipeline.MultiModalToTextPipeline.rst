mindformers.pipeline.MultiModalToTextPipeline
===============================================

.. py:class:: mindformers.pipeline.MultiModalToTextPipeline(model: Union[PreTrainedModel, Model], processor: Optional[BaseXModalToTextProcessor] = None, **kwargs)

    多模态文本生成的推理流程。

    参数：
        - **model** (Union[PretrainedModel, Model]) - 执行任务的模型。必须是继承自 `PretrainedModel` 类的模型实例。
        - **processor** (BaseXModalToTextProcessor, 可选) - 模型的图片处理器。默认值： ``None`` 。
    
    返回：
        一个 `MultiModalToTextPipeline` 实例。

    异常：
        - **TypeError** - 如果输入模型和图片处理流程的类型设置错误。
        - **ValueError** - 如果输入模型不在支持列表中。
