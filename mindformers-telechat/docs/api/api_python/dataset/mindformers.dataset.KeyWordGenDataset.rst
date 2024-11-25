mindformers.dataset.KeyWordGenDataset
=====================================

.. py:class:: mindformers.dataset.KeyWordGenDataset(dataset_config: Optional[dict] = None, data_loader: Union[dict, Callable] = None, tokenizer: Union[dict, Callable] = None, input_columns: List[str] = None, batch_size: int = 8, drop_remainder: bool = True, num_parallel_workers: int = 8, repeat: int = 1, ignore_pad_token_for_loss: bool = True, max_source_length: int = None, max_target_length: int = None, phase: str = 'train', version: int = 1, seed: int = 0, prefetch_size: int = 1, numa_enable: bool = False, auto_tune: bool = False, filepath_prefix: str = './autotune', autotune_per_step: int = 10, profile: bool = False, **kwargs)

    关键词生成数据集。
    根据不同的 `phase` 配置，数据集会生成不同的输出列。

    - `phase` 为 ``'train'`` ，输出列为 `[input_ids, labels, position_ids, attention_mask]` 。
    - `phase` 为 ``'eval'`` ，输出列为 `[input_ids, labels]` 。

    输出列均会被转换为int32类型。

    参数：
        - **dataset_config** (dict, 可选) - 数据集配置信息。当传入的 `dataset_config` 为空字典或 ``None`` 时， `dataset_config` 将由以下所有参数生成；否则以下所有参数被忽略。默认值： ``None`` 。
        - **data_loader** (Union[dict, Callable]) - 必须是包含data loader配置信息的字典，或一个data loader实例。当 `data_loader` 为 `dict` 类型时，字典的键可以是"type"、"dataset_dir"、"dataset_files"、"phase"、"shuffle"、"origin_columns"和"version"。

          - ``"type"`` - 必选。数据集的类型。必须是 `str` 或 `type` 类型。当 ``"type"`` 对应值为"MindDataset"时， ``"dataset_dir"`` 与 ``"dataset_files"`` 中必选两者之一，优先使用 ``"dataset_dir"`` ；否则必选 ``"dataset_dir"`` 。
          - ``"dataset_dir"`` - 数据集文件所在路径或目录。当 ``"type"`` 为"MindDataset"且 ``"dataset_dir"`` 表示一个目录时，将递归查找目录下所有 `mindrecord` 格式文件。
          - ``"dataset_files"`` - `mindrecord` 格式文件所在路径。当 ``"type"`` 为"MindDataset"时生效；否则键被忽略。必须是 `list` 或 `tuple` 类型。
          - ``"phase"`` - 必选。需要读取的数据集的子集，可选值为"train"和"eval"。
          - ``"shuffle"`` - 必选。指示是否混洗数据集。必须是 `bool` 类型。
          - ``"origin_columns"`` - 必选。表示"prompt"和"answer"在数据文件中的对应列名。必须是两个字符串组成的列表。
          - ``"version"`` - 可选。映射函数的版本，可选值为"1"和"2"。未配置键时，默认为 ``1`` 。

        - **tokenizer** (Union[dict, Callable]) - 必须是包含分词器配置信息的字典，或一个分词器实例。
        - **input_columns** (list[str]) - 表示映射处理前的数据列名称。
        - **batch_size** (int) - 每个批次的大小。默认值： ``8`` 。
        - **drop_remainder** (bool) - 是否在最后一个批次的数据项数小于批次大小时，丢弃最后一个批次。默认值： ``True`` 。
        - **num_parallel_workers** (int) - 并行执行数据映射处理的进程/线程数。默认值： ``8`` 。
        - **repeat** (int) - 数据集重复的次数。默认值： ``1`` 。
        - **ignore_pad_token_for_loss** (bool) - 是否忽略<pad>词元对应的损失。默认值：``True``。
        - **max_source_length** (int) - 源序列的最大长度。
        - **max_target_length** (int) - 目标序列的最大长度。
        - **phase** (int) - 需要读取的数据集的子集，`data_loader` 为 `dict` 类型时忽略该参数。可选值为 'train' 或 'eval'。默认值： ``'train'``。
        - **version** (int) - 映射函数的版本， `data_loader` 为 `dict` 类型时忽略该参数。可选值为 `1` 或 `2`。默认值：``1``。
        - **seed** (int) - 随机数种子。默认值： ``0`` 。
        - **prefetch_size** (int) - 流水线中每个数据处理操作的缓存队列大小。默认值： ``1`` 。
        - **numa_enable** (bool) - 是否采用NUMA绑定函数。默认值： ``False`` 。
        - **auto_tune** (bool) - 是否启用数据处理参数自动优化。默认值： ``False`` 。
        - **autotune_per_step** (int) - 设置调整自动数据加速配置步骤的间隔。默认值： ``10`` 。
        - **filepath_prefix** (str) - 保存优化参数配置的路径。默认值： ``'./autotune'`` 。
        - **profile** (bool) - 是否启用数据收集。默认值： ``False`` 。

    返回：
        `KeyWordGenDataset` 实例。

    异常：
        - **ValueError** -  `dataset_config.data_loader` 中没有指定键 `"dataset_dir"` 或 `"dataset_files"` 。
