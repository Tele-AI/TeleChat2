mindformers.pet.pet_config.LoraConfig
=====================================

.. py:class:: mindformers.pet.pet_config.LoraConfig(lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.01, lora_a_init: str = 'normal', lora_b_init: str = 'zero', param_init_type: str = 'float16', compute_dtype: str = 'float16', target_modules: str = None, exclude_layers: str = None, freeze_include: List[str] = None, freeze_exclude: List[str] = None, **kwargs)

    LoRA算法的配置信息，用于设置LoRA模型运行时的参数。

    参数：
        - **lora_rank** (int, 可选) - LoRA矩阵的行(列)数。默认值： ``8`` 。
        - **lora_alpha** (int, 可选) - lora_rank中的一个常量。默认值： ``16`` 。
        - **lora_dropout** (float, 可选) - 丢弃率，大于等于0且小于1。默认值： ``0.01``。
        - **lora_a_init** (str, 可选) - LoRA A矩阵的初始化策略。参考 (`LoRA A矩阵的初始化策略 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_)。默认值： ``normal``。
        - **lora_b_init** (str, 可选) - LoRA B矩阵的初始化策略。参考 (`LoRA B矩阵的初始化策略 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_)。默认值： ``zero``。
        - **param_init_type** (str, 可选) - 初始化张量中的数据类型。默认值： ``float16``。
        - **compute_dtype** (str, 可选) - 数据的计算类型。默认值： ``float16``。
        - **target_modules** (str, 可选) - 需要使用LoRA算法替换的层。默认值： ``None``。
        - **exclude_layers** (str, 可选) - 不需要使用LoRA算法替换的层。默认值： ``None``。
        - **freeze_include** (List[str], 可选) - 待冻结的模块列表。默认值： ``None``。
        - **freeze_exclude** (List[str], 可选) - 不需要冻结的模块列表。当freeze_include和freeze_exclude中的某一项冲突时，匹配该项的模块将不会被处理。默认值： ``None``。

    返回：
        LoraConfig实例。