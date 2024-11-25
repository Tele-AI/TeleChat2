mindformers.pet.models.LoraModel
================================

.. py:class:: mindformers.pet.models.LoraModel(config: LoraConfig, base_model: PreTrainedModel)

    LLM的LoRA模型。提供了一种灵活且高效的方式来调整和优化预训练模型，为基础预训练模型添加LoRA结构。

    参数：
        - **config** (LoraConfig) - 低参微调（Pet）算法的配置基类。
        - **base_model** (PreTrainedModel) - 用于调优的预训练模型。

    返回：
        LoraModel实例。