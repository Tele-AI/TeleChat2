mindformers.dataset.ContrastiveLanguageImagePretrainDataset
===========================================================

.. py:class:: mindformers.dataset.ContrastiveLanguageImagePretrainDataset(dataset_config: Optional[dict] = None, data_loader: Union[dict, Callable] = None, transforms: Union[dict, list] = None, text_transforms: Union[dict, list] = None, tokenizer: Union[dict, Callable] = None, sampler: Union[dict, Callable] = None, batch_size: int = 8, drop_remainder: bool = True, num_parallel_workers: int = 8, python_multiprocessing: bool = False, repeat: int = 1, seed: int = 0, prefetch_size: int = 1, numa_enable: bool = False, auto_tune: bool = False, filepath_prefix: str = './autotune', autotune_per_step: int = 10, profile: bool = False, **kwargs)

    CLIP（Contrastive Language-Image Pre-training）文图对比预训练数据集。
    生成的数据集有两列 `[image, text]` ，数据列的类型取决于读取的数据集的数据格式和采用的数据变换操作。

    参数：
        - **dataset_config** (dict, 可选) - 数据集配置信息。当传入的 `dataset_config` 为空字典或 ``None`` 时， `dataset_config` 将由以下所有参数生成；否则以下所有参数被忽略。默认值： ``None`` 。
        - **data_loader** (Union[dict, Callable]) - 必须是包含data loader配置信息的字典，或一个data loader实例。当 `dataset_config.data_loader` 为 `dict` 类型时，字典的键可以是"type"、"dataset_dir"、"stage"和"column_names"。

          - ``"type"`` - 必选。数据集的类型。必须是 `str` 或 `type` 类型。
          - ``"dataset_dir"`` - 必选。数据集文件所在目录。
          - ``"stage"`` - 可选。需要读取的数据集的子集，可选值为"train"、"test"、"dev"和"all"。未配置键时，默认为 ``"train"`` 。
          - ``"column_names"`` - 可选。数据集的数据列名。必须是由字符串组成的 `list` 或 `tuple` 类型。未配置键时，默认为 ``["image", "text"]`` 。

        - **transforms** (Union[dict, list]) - 必须是由图像变换操作配置信息或实例组成的字典或列表。默认值： ``None`` ，表示不进行图像变换。
        - **text_transforms** (Union[dict, list]) - 必须是由文本变换操作配置信息或实例组成的字典或列表。默认值： ``None`` ，表示不进行文本变换。
        - **tokenizer** (Union[dict, Callable]) - 必须是包含分词器配置信息的字典，或一个分词器实例。默认值： ``None`` ，表示不使用分词器。
        - **sampler** (Union[dict, Callable]) - 必须是包含采样器配置信息的字典，或一个采样器实例。默认值： ``None`` ，表示不使用采样器。
        - **batch_size** (int) - 每个批次的大小。默认值： ``8`` 。
        - **drop_remainder** (bool) - 是否在最后一个批次的数据项数小于批次大小时，丢弃最后一个批次。默认值： ``True`` 。
        - **num_parallel_workers** (int) - 并行执行数据映射处理的进程/线程数。默认值： ``8`` 。
        - **python_multiprocessing** (bool) - 是否启用Python的Multi-Process模块以加速映射操作。默认值：``False`` 。
        - **repeat** (int) - 数据集重复的次数。默认值： ``1`` 。
        - **seed** (int) - 随机数种子。默认值： ``0`` 。
        - **prefetch_size** (int) - 流水线中每个数据处理操作的缓存队列大小。默认值： ``1`` 。
        - **numa_enable** (bool) - 是否采用NUMA绑定函数。默认值： ``False`` 。
        - **auto_tune** (bool) - 是否启用数据处理参数自动优化。默认值： ``False`` 。
        - **autotune_per_step** (int) - 设置调整自动数据加速配置步骤的间隔。默认值： ``10`` 。
        - **filepath_prefix** (str) - 保存优化参数配置的路径。默认值： ``'./autotune'`` 。
        - **profile** (bool) - 是否启用数据收集。默认值： ``False`` 。

    返回：
        `ContrastiveLanguageImagePretrainDataset` 实例。
