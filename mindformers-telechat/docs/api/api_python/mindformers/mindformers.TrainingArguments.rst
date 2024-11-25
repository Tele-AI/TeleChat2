mindformers.TrainingArguments
==============================

.. py:class:: mindformers.TrainingArguments(output_dir = './output', overwrite_output_dir = False, seed = 42, data_seed = None, only_save_strategy = False, auto_trans_ckpt = False, src_strategy = None, transform_process_num = 1, resume_from_checkpoint = None, resume_training = None, ignore_data_skip = False, data_skip_steps = None, do_train = False, do_eval = False, do_predict = False, check_for_nan_in_loss_and_grad = False, calculate_per_token_loss = False, remote_save_url = None, batch_size = None, num_train_epochs = 3.0, sink_mode = True, sink_size = 2, gradient_accumulation_steps = 1, mode = 0, use_cpu = False, device_id = 0, device_target = 'Ascend', enable_graph_kernel = False, max_call_depth = 10000, max_device_memory = '1024GB', save_graphs = False, save_graphs_path = './graph', use_parallel = False, parallel_mode = 1, gradients_mean = False, loss_repeated_mean = False, enable_alltoall = False, full_batch = True, dataset_strategy = 'full_batch', search_mode = 'sharding_propagation', enable_parallel_optimizer = False, gradient_accumulation_shard = False, parallel_optimizer_threshold = 64, optimizer_weight_shard_size = -1, strategy_ckpt_save_file = './ckpt_strategy.ckpt', data_parallel = 1, model_parallel = 1, expert_parallel = 1, pipeline_stage = 1, micro_batch_num = 1, gradient_aggregation_group = 4, micro_batch_interleave_num = 1, use_seq_parallel = False, vocab_emb_dp = True, expert_num = 1, capacity_factor = 1.05, aux_loss_factor = 0.05, num_experts_chosen = 1, recompute = False, select_recompute = False, parallel_optimizer_comm_recompute = False, mp_comm_recompute = True, recompute_slice_activation = False, optim = 'fp32_adamw', adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-08, weight_decay = 0.0, layer_scale = False, layer_decay = 0.65, lr_scheduler_type = 'cosine', learning_rate = 5e-05, lr_end = 1e-06, warmup_lr_init = 0.0, warmup_epochs = None, warmup_ratio = None, warmup_steps = 0, total_steps = -1, lr_scale = False, lr_scale_factor = 256, dataset_task = None, dataset_type = None, train_dataset = None, train_dataset_in_columns = None, train_dataset_out_columns = None, eval_dataset = None, eval_dataset_in_columns = None, eval_dataset_out_columns = None, shuffle = True, dataloader_drop_last = True, repeat = 1, per_device_train_batch_size = 8, per_device_eval_batch_size = 8, dataloader_num_workers = 8, python_multiprocessing = False, numa_enable = False, prefetch_size = 1, wrapper_type = 'MFTrainOneStepCell', scale_sense = 'DynamicLossScaleUpdateCell', loss_scale_value = 65536, loss_scale_factor = 2, loss_scale_window = 1000, use_clip_grad = True, max_grad_norm = 1.0, max_scale_window = 1000, min_scale_window = 20, metric_type = None, logging_strategy = 'steps', logging_steps = 1, save_prefix = 'CKP', save_directory = None, save_strategy = 'steps', save_steps = 500, save_seconds = None, save_total_limit = 5, keep_checkpoint_per_n_minutes = 0, save_on_each_node = True, integrated_save = None, save_network_params = True, save_trainable_params = False, async_save = False, evaluation_strategy = 'no', eval_steps = None, eval_epochs = None, profile = False, profile_start_step = 1, profile_end_step = 10, init_start_profile = False, profile_communication = False, profile_memory = True, auto_tune = False, filepath_prefix = './autotune', autotune_per_step = 10, push_to_hub = False, hub_model_id = None, hub_strategy = 'every_save', hub_token = None, hub_private_repo = False, hub_always_push = False)

    TrainingArguments是与MindSpore训练相关的参数集合。

    参数：
        - **output_dir** (str, 可选) - checkpoint和log文件保存的输出目录。默认值： ``"./output"`` 。
        - **overwrite_output_dir** (bool, 可选) - 是否覆盖输出目录中的内容。如果 `output_dir` 指向的是checkpoint文件的话，该字段用于恢复训练。默认值： ``False`` 。
        - **seed** (int, 可选) - 训练任务的随机数种子。默认值： ``42`` 。
        - **data_seed** (int, 可选) - 数据采样的随机数种子。默认值： ``None`` 。
        - **only_save_strategy** (bool, 可选) - 如果为True时，任务会把策略文件保存到 `output_dir/strategy` 目录。只有当 `use_parallel` 为True时才生效。默认值： ``False`` 。
        - **auto_trans_ckpt** (bool, 可选) - 是否根据并行配置自动转换checkpoint。默认值： ``False`` 。
        - **src_strategy** (str, 可选) - 代表权重转换时的策略文件，只有 `auto_trans_ckpt` 为True时生效。默认值： ``None`` 。
        - **transform_process_num** (int, 可选) - 转换checkpoint的进程数。默认值： ``1`` 。
        - **resume_from_checkpoint** (Union[str, bool], 可选) - 模型的checkpoint文件夹路径。默认值： ``None`` 。
        - **resume_training** (Union[bool, str], 可选) - 指定是否恢复训练，或者指定用于恢复训练的checkpoint名称。默认值： ``None`` 。
        - **ignore_data_skip** (bool, 可选) - 在恢复训练时，是否跳过执行过的批次，加载与前一次训练相同阶段的数据。默认值： ``False`` 。
        - **data_skip_steps** (int, 可选) - 在恢复训练时，指定数据集中跳过的步数。只有在 `ignore_data_skip` 为False时生效。默认值： ``None`` 。
        - **do_train** (bool, 可选) - 是否执行训练。默认值： ``False`` 。
        - **do_eval** (bool, 可选) - 是否执行评估。默认值： ``False`` 。
        - **do_predict** (bool, 可选) - 是否执行预测。默认值： ``False`` 。
        - **check_for_nan_in_loss_and_grad** (bool, 可选) - 是否在训练中检查损失和梯度存在Nan。默认值： ``False`` 。
        - **calculate_per_token_loss** (bool, 可选) - 是否计算每个token的损失。默认值： ``False`` 。
        - **remote_save_url** (str, 可选) - 在ModeArts上执行训练任务时的OBS输出路径。默认值： ``None`` 。
        - **batch_size** (int, 可选) - 输入数据的batch size，如果设置了值，则会覆盖 `per_device_train_batch_size` 。默认值： ``None`` 。
        - **num_train_epochs** (float, 可选) - 训练任务的epoch总数。默认值： ``3.0`` 。
        - **sink_mode** (bool, 可选) - 是否直接下沉数据到设备端。默认值： ``True`` 。
        - **sink_size** (int, 可选) - 训练或评估时，每一步下沉的数据量。默认值： ``2`` 。
        - **gradient_accumulation_steps** (int, 可选) - 在执行反向传播前，累积梯度的步数。默认值： ``1`` 。
        - **mode** (int, 可选) - 运行模式，包括GRAPH_MODE(0)或者PYNATIVE_MODE(1)。默认值： ``0`` 。
        - **use_cpu** (bool, 可选) - 是否使用cpu。默认值： ``False`` 。
        - **device_id** (int, 可选) - 默认的设备id。默认值： ``0`` 。
        - **device_target** (str, 可选) - 默认的目标设备，支持'Ascend', 'GPU'和'CPU'。默认值： ``"Ascend"`` 。
        - **enable_graph_kernel** (bool, 可选) - 是否支持图融合。默认值： ``False`` 。
        - **max_call_depth** (int, 可选) - 最大函数调用深度。默认值： ``10000`` 。
        - **max_device_memory** (str, 可选) - 设备端的最大可用内存。默认值： ``"1024GB"`` 。
        - **save_graphs** (bool, 可选) - 是否保存中间编译的图。默认值： ``False`` 。
        - **save_graphs_path** (str, 可选) - 保存中间编译的图的路径。默认值： ``"./graph"`` 。
        - **use_parallel** (bool, 可选) - 是否对神经网络开启分布式并行。默认值： ``False`` 。
        - **parallel_mode** (int, 可选) - 并行模式，包括数据并行(0)，半自动并行(1)，全自动并行(2)和混合并行(3)。默认值： ``1`` 。
        - **gradients_mean** (bool, 可选) - 在梯度汇总后，是否使用平均值算子。通常，在半自动并行模式下值为False，在数据并行模式下值为True。默认值： ``False`` 。
        - **loss_repeated_mean** (bool, 可选) - 表示在重复计算时，是否向后执行均值操作符。默认值： ``False`` 。
        - **enable_alltoall** (bool, 可选) - 在通信中，是否允许使用AllToAll通信算子。默认值： ``False`` 。
        - **full_batch** (bool, 可选) - 如果在自动并行模式下加载了真个批次的数据集，那么 `full_batch` 应该置为True。当前更推荐使用 `dataset_strategy` 。默认值： ``True`` 。
        - **dataset_strategy** (Union[str, tuple], 可选) - 数据集切分策略，半自动并行模式下设置为"full_batch"，数据并行模式下设置为"data_parallel"。默认值： ``"full_batch"`` 。
        - **search_mode** (str, 可选) - 策略搜索模式，只有在自动并行模式下生效。默认值： ``"sharding_propagation"`` 。
        - **enable_parallel_optimizer** (bool, 可选) - 是否开启优化器并行。默认值： ``False`` 。
        - **gradient_accumulation_shard** (bool, 可选) - 是否对累积的梯度在数据并行维度进行切分。可以降低对内存的消耗，但是会导致在反向传播时增加额外的ReduceScatter通信。默认值： ``False`` 。
        - **parallel_optimizer_threshold** (int, 可选) - 参数切分的阈值。默认值： ``64`` 。
        - **optimizer_weight_shard_size** (int, 可选) - 对特定的优化器权重进行切分时的通信域大小。默认值： ``1`` 。
        - **strategy_ckpt_save_file** (str, 可选) - 分布式策略文件保存的路径。默认值： ``"./ckpt_strategy.ckpt"`` 。
        - **data_parallel** (int, 可选) - 数据并行的切分数量。默认值： ``1`` 。
        - **model_parallel** (int, 可选) - 模型并行的切分数量。默认值： ``1`` 。
        - **expert_parallel** (int, 可选) - 专家并行的切分数量。默认值： ``1`` 。
        - **pipeline_stage** (int, 可选) - 流水并行的切分数量。默认值： ``1`` 。
        - **micro_batch_num** (int, 可选) - 微批次的大小。只有 `pipeline_stage` 大于1时生效。默认值： ``1`` 。
        - **gradient_aggregation_group** (int, 可选) - 梯度通信算子融合组的大小。默认值： ``4`` 。
        - **micro_batch_interleave_num** (int, 可选) - 在 `micro_batch_interleave_num` 大于1时使能多副本并行。默认值： ``1`` 。
        - **use_seq_parallel** (bool, 可选) - 是否开启序列并行。默认值： ``False`` 。
        - **vocab_emb_dp** (bool, 可选) - 是否旨在数据并行维度上切分词汇。默认值： ``True`` 。
        - **expert_num** (int, 可选) - 专家并行中专家的数量。默认值： ``1`` 。
        - **capacity_factor** (float, 可选) - 容量因子。默认值： ``1.05`` 。
        - **aux_loss_factor** (float, 可选) - 损失(loss)贡献因子。默认值： ``0.05`` 。
        - **num_experts_chosen** (int, 可选) - 对每个token最多选择专家的数量。默认值： ``1`` 。
        - **recompute** (bool, 可选) - 是否开启重计算。默认值： ``False`` 。
        - **select_recompute** (bool, 可选) - 是否开启选择重计算。默认值： ``False`` 。
        - **parallel_optimizer_comm_recompute** (bool, 可选) - 在优化器并行下，是否重计算AllGather通信。默认值： ``False`` 。
        - **mp_comm_recompute** (bool, 可选) - 在模型并行下，是否重计算通信算子。默认值： ``True`` 。
        - **recompute_slice_activation** (bool, 可选) - 是否对保存在内存中的输出结果进行切片。默认值： ``False`` 。
        - **optim** (Union[OptimizerType, str], 可选) - 优化器类型。默认值： ``"fp32_adamw"`` 。
        - **adam_beta1** (float, 可选) - AdamW优化器的Beta1参数。默认值： ``0.9`` 。
        - **adam_beta2** (float, 可选) - AdamW优化器的Beta2参数。默认值： ``0.999`` 。
        - **adam_epsilon** (float, 可选) - AdamW优化器的Epsilon参数。默认值： ``1.e-8`` 。
        - **weight_decay** (float, 可选) - AdamW优化器的权重衰减参数。默认值： ``0.0`` 。
        - **layer_scale** (bool, 可选) - 是否开启按层衰减。默认值： ``False`` 。
        - **layer_decay** (float, 可选) - 层衰减的系数。默认值： ``0.65`` 。
        - **lr_scheduler_type** (Union[LrSchedulerType, str], 可选) - 学习率调度器类型。默认值： ``"cosine"`` 。
        - **learning_rate** (float, 可选) - 初始学习率。默认值： ``5.e-5`` 。
        - **lr_end** (float, 可选) - 学习率最终值。默认值： ``1.e-6`` 。
        - **warmup_lr_init** (float, 可选) - 学习率预热的初始值。默认值： ``0.0`` 。
        - **warmup_epochs** (int, 可选) - 线性预热的epoch数量。默认值： ``None`` 。
        - **warmup_ratio** (float, 可选) - 线性预热的步数占总步数的比例。默认值： ``None`` 。
        - **warmup_steps** (int, 可选) - 线性预热的步数。默认值： ``0`` 。
        - **total_steps** (int, 可选) - 用于计算学习率的总步数。默认值： ``-1`` 。
        - **lr_scale** (bool, 可选) - 是否开启学习率缩放。默认值： ``False`` 。
        - **lr_scale_factor** (int, 可选) - 学习率缩放因子。默认值： ``256`` 。
        - **dataset_task** (str, 可选) - 数据集任务名称。默认值： ``None`` 。
        - **dataset_type** (str, 可选) - 数据集类型。默认值： ``None`` 。
        - **train_dataset** (str, 可选) - 训练数据集路径。默认值： ``None`` 。
        - **train_dataset_in_columns** (List[str], 可选) - 训练数据集的输入列。默认值： ``None`` 。
        - **train_dataset_out_columns** (List[str], 可选) - 训练数据集的输出列。默认值： ``None`` 。
        - **eval_dataset** (str, 可选) - 评估数据集。默认值： ``None`` 。
        - **eval_dataset_in_columns** (List[str], 可选) - 评估数据集的输入列。默认值： ``None`` 。
        - **eval_dataset_out_columns** (List[str], 可选) - 评估数据集的输出列。默认值： ``None`` 。
        - **shuffle** (bool, 可选) - 是否对训练数据集打散。默认值： ``True`` 。
        - **dataloader_drop_last** (bool, 可选) - 是否丢弃最后一个大小不能被batch size整除的批次。默认值： ``True`` 。
        - **repeat** (int, 可选) - 数据集重复的次数。默认值： ``1`` 。
        - **per_device_train_batch_size** (int, 可选) - 每个设备的训练数据集的batch size。默认值： ``8`` 。
        - **per_device_eval_batch_size** (int, 可选) - 每个设备的评估数据集的batch size。默认值： ``8`` 。
        - **dataloader_num_workers** (int, 可选) - 加载数据集的进程数量。默认值： ``8`` 。
        - **python_multiprocessing** (bool, 可选) - 是否开启python的多进程模式。默认值： ``False`` 。
        - **numa_enable** (bool, 可选) - 设置NUMA的默认状态。默认值： ``False`` 。
        - **prefetch_size** (int, 可选) - 设置线程队列的容量。默认值： ``1`` 。
        - **wrapper_type** (str, 可选) - 装饰器的类型。默认值： ``"MFTrainOneStepCell"`` 。
        - **scale_sense** (Union[str, float], 可选) - 设置损失(loss)缩放的类。默认值： ``"DynamicLossScaleUpdateCell"`` 。
        - **loss_scale_value** (int, 可选) - 设置损失(loss)缩放的因子。默认值： ``65536`` 。
        - **loss_scale_factor** (int, 可选) - 设置损失(loss)缩放系数的递增或递减因子。默认值： ``2`` 。
        - **loss_scale_window** (int, 可选) - 增加损失(loss)缩放系数的最大连续训练的步数。默认值： ``1000`` 。
        - **use_clip_grad** (bool, 可选) - 是否开启梯度裁剪。默认值： ``True`` 。
        - **max_grad_norm** (float, 可选) - 最大梯度规范化的值。默认值： ``1.0`` 。
        - **max_scale_window** (int, 可选) - 最大缩放窗口值。默认值： ``1000`` 。
        - **min_scale_window** (int, 可选) - 最小缩放窗口值。默认值： ``20`` 。
        - **metric_type** (Union[List[str], str], 可选) - 指标类型。默认值： ``None`` 。
        - **logging_strategy** (Union[LoggingIntervalStrategy, str], 可选) - 日志策略。默认值： ``"steps"`` 。
        - **logging_steps** (int, 可选) - 记录日志的间隔步数。默认值： ``1`` 。
        - **save_prefix** (str, 可选) - checkpoint文件名称的前缀。默认值： ``"CKP"`` 。
        - **save_directory** (str, 可选) - 保存checkpoint文件的目录。默认值： ``None`` 。
        - **save_strategy** (Union[SaveIntervalStrategy, str], 可选) - checkpoint的保存策略。默认值： ``"steps"`` 。
        - **save_steps** (int, 可选) - 保存checkpoint文件的间隔步数。默认值： ``500`` 。
        - **save_seconds** (int, 可选) - 保存checkpoint文件间隔的时间（单位：秒）。默认值： ``None`` 。
        - **save_total_limit** (int, 可选) - 最多保存checkpoint文件的数量。默认值： ``5`` 。
        - **keep_checkpoint_per_n_minutes** (int, 可选) - 间隔多少分钟保存一次checkpoint。默认值： ``0`` 。
        - **save_on_each_node** (bool, 可选) - 多节点分布式训练时，是否在每个节点都保存checkpoint。默认值： ``True`` 。
        - **integrated_save** (bool, 可选) - 是否合并并且保存被分割的张量。默认值： ``None`` 。
        - **save_network_params** (bool, 可选) - 是否保存网络权重参数。默认值： ``True`` 。
        - **save_trainable_params** (bool, 可选) - 是否保存微调参数。默认值： ``False`` 。
        - **async_save** (bool, 可选) - 是否异步保存checkpoint。默认值： ``False`` 。
        - **evaluation_strategy** (Union[IntervalStrategy, str], 可选) - 评估策略。默认值： ``"no"`` 。
        - **eval_steps** (float, 可选) - 执行评估的间隔步数。默认值： ``None`` 。
        - **eval_epochs** (int, 可选) - 执行评估的间隔epoch数量。默认值： ``None`` 。
        - **profile** (bool, 可选) - 是否开启性能分析工具。默认值： ``False`` 。
        - **profile_start_step** (int, 可选) - 在第几步开启性能分析。默认值： ``1`` 。
        - **profile_end_step** (int, 可选) - 在第几步结束性能分析。默认值： ``10`` 。
        - **init_start_profile** (bool, 可选) - 在性能分析初始化时是否采集数据。默认值： ``False`` 。
        - **profile_communication** (bool, 可选) - 在多卡训练时，是否开启通信性能数据采集。
        - **profile_memory** (bool, 可选) - 是否采集张量内存数据。默认值： ``True`` 。
        - **auto_tune** (bool, 可选) - 是否开启自动数据加速。默认值： ``False`` 。
        - **filepath_prefix** (str, 可选) - 经过优化的全局配置的保存路径和文件前缀。默认值： ``"./autotune"`` 。
        - **autotune_per_step** (int, 可选) - 调整自动数据加速配置的间隔步数。默认值： ``10`` 。
        - **push_to_hub** (bool, 可选) - 是否上传模型。默认值： ``False`` 。
        - **hub_model_id** (str, 可选) - 保存模型的仓库名称。默认值： ``None`` 。
        - **hub_strategy** (Union[HubStrategy, str], 可选) - 上传模型的策略。默认值： ``"every_save"`` 。
        - **hub_token** (str, 可选) - 推送模型时的token。默认值： ``None`` 。
        - **hub_private_repo** (bool, 可选) - 模型仓库是否是私有的。默认值： ``False`` 。
        - **hub_always_push** (bool, 可选) - 在值不为True时，如果前一次上传尚未完成，会跳过推送操作。默认值： ``False`` 。

    返回：
        TrainingArguments类的实例。

    .. py:method:: convert_args_to_mindformers_config(task_config: MindFormerConfig = None)

        把训练参数转换成MindFormers的config类型。

        参数：
            - **task_config** (MindFormerConfig, 可选) - 任务配置信息。

        返回：
            MindFormerConfig类的实例，包含经过处理的任务配置信息。

    .. py:method:: get_moe_config()

        获取moe配置。

        返回：
            MoEConfig实例。

    .. py:method:: get_parallel_config()

        获取并行配置。

        返回：
            TransformerOpParallelConfig实例。

    .. py:method:: get_recompute_config()

        获取重计算配置。

        返回：
            TransformerRecomputeConfig实例。

    .. py:method:: get_warmup_steps(num_training_steps: int)

        获取线性预热阶段的步数。

        参数：
            - **num_training_steps** (int) - 训练步数。

        返回：
            warmup_steps的值，即预热阶段步数。

    .. py:method:: set_dataloader(train_batch_size: int = 8, eval_batch_size: int = 8, drop_last: bool = False, num_workers: int = 0, ignore_data_skip: bool = False, data_skip_steps: Optional[int] = None, sampler_seed: Optional[int] = None, **kwargs)

        设置与创建dataloader相关的参数。

        参数：
            - **train_batch_size** (int, 可选) - 训练过程中数据集的batch size。默认值： ``8`` 。
            - **eval_batch_size** (int, 可选) - 评估过程中数据集的batch size。默认值： ``8`` 。
            - **drop_last** (bool, 可选) - 是否丢弃最后一个不完整的batch（如果数据集长度不能被batch size整除的话）。默认值： ``False`` 。
            - **num_workers** (int, 可选) - 数据集加载的进程数，0意味着通过主进程来进行加载。默认值： ``0`` 。
            - **ignore_data_skip** (bool, 可选) - 在恢复训练时，是否跳过数据集已经处理过的批次，从而加载前一次训练相同步骤的数据。默认值： ``False`` 。
            - **data_skip_steps** (int, 可选) - 在恢复训练时，指定在训练数据集中跳过的步数。只有 `ignore_data_skip` 值为False时生效。默认值： ``None`` 。
            - **sampler_seed** (int, 可选) - 数据采样中的随机数种子。如果未设置，用于数据采样的随机生成器将使用与 `self.seed` 相同的种子。这可用于确保数据采样的可重复性，独立于模型的seed。默认值： ``None`` 。
            - **kwargs** (Any) - 其它参数。

    .. py:method:: set_logging(strategy: Union[str, IntervalStrategy] = 'steps', steps: int = 500, **kwargs)

        设置与日志相关的参数。

        参数：
            - **strategy** (Union[str, IntervalStrategy], 可选) - 训练过程中记录日志的策略，"no"表示训练中不记录日志，"epoch"表示训练中每个epoch结束后记录日志，"steps"表示训练中每经过 `steps` 步数后记录日志。默认值： ``"steps"`` 。
            - **steps** (int, 可选) - 两次日志之间间隔的步数，在 `strategy` 值为 `steps` 时生效。默认值： ``500`` 。
            - **kwargs** (Any) - 其它参数。

    .. py:method:: set_lr_scheduler(name: Union[str, LrSchedulerType] = 'linear', num_epochs: float = 3.0, warmup_lr_init: float = 0.0, warmup_epochs: Optional[int] = None, warmup_ratio: Optional[float] = None, warmup_steps: int = 0, total_steps: int = - 1, **kwargs)

        设置与学习率调度器相关的参数。

        参数：
            - **name** (Union[str, LrSchedulerType], 可选) - 使用的调度器类型。默认值： ``"linear"`` 。
            - **num_epochs** (float, 可选) - 训练执行的epoch数量。默认值： ``3.0`` 。
            - **warmup_lr_init** (float, 可选) - 学习率预热的起始值。默认值： ``0.0`` 。
            - **warmup_epochs** (int, 可选) - 预热的epoch数量。默认值： ``None`` 。
            - **warmup_ratio** (float, 可选) - 预热阶段的步数占总训练步数的比例。默认值： ``None`` 。
            - **warmup_steps** (int, 可选) - 预热阶段的步数，如果同时设置了warmup_steps和warmup_ratio，则使用warmup_steps。默认值： ``0`` 。
            - **total_steps** (int, 可选) - 总步数。默认值： ``-1`` 。
            - **kwargs** (Any) - 其它参数。

    .. py:method:: set_optimizer(name: Union[str, OptimizerType] = 'adamw', learning_rate: float = 5e-5, lr_end: float = 1e-6, weight_decay: float = 0, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, **kwargs)

        设置与优化器相关的参数。

        参数：
            - **name** (Union[str, OptimizerType], 可选) - 使用的优化器种类。默认值： ``"adamw"`` 。
            - **learning_rate** (float, 可选) - 初始的学习率。默认值： ``5e-5`` 。
            - **lr_end** (float, 可选) - 最终的学习率。默认值： ``1e-6`` 。
            - **weight_decay** (float, 可选) - 不为0时，用户神经网络所有层（bias和LayerNorm权重除外）的权重衰减。默认值： ``0`` 。
            - **beta1** (float, 可选) - adam优化器的beta1超参。默认值： ``0.9`` 。
            - **beta2** (float, 可选) - adam优化器的beta2超参。默认值： ``0.999`` 。
            - **epsilon** (float, 可选) - adam优化器的epsilon超参。默认值： ``1e-8`` 。
            - **kwargs** (Any) - 其它参数。

    .. py:method:: set_save(strategy: Union[str, IntervalStrategy] = 'steps', steps: int = 500, total_limit: Optional[int] = None, on_each_node: bool = True, **kwargs)

        设置与checkpoint保存相关的参数。

        参数：
            - **strategy** (Union[str, IntervalStrategy], 可选) - 训练过程中保存权重的策略，"no"表示训练中不保存权重，"epoch"表示训练中每个epoch结束后保存权重，"steps"表示训练中每经过 `steps` 步数后保存权重。默认值： ``"steps"`` 。
            - **steps** (int, 可选) - 两次保存权重之间间隔的步数，在 `strategy` 值为 `steps` 时生效。默认值： ``500`` 。
            - **total_limit** (int, 可选) - checkpoint的总数量，如果超过该数量，会删除 `output_dir` 目录下最老的权重。默认值： ``None`` 。
            - **on_each_node** (bool, 可选) - 在多节点分布式训练时，控制在每个节点上保存权重或者只在主节点上保存。默认值： ``True`` 。
            - **kwargs** (Any) - 其它参数。

    .. py:method:: set_training(learning_rate: float = 5e-5, batch_size: int = 8, weight_decay: float = 0, num_epochs: float = 3.0, gradient_accumulation_steps: int = 1, seed: int = 42, **kwargs)

        设置与训练相关的所有参数。调用该方法时候会自动设置 `self.do_train` 为True。

        参数：
            - **learning_rate** (float, 可选) - 优化器的初始学习率。默认值： ``5e-5`` 。
            - **batch_size** (int, 可选) - 训练过程中数据集的batch size。默认值： ``8`` 。
            - **weight_decay** (float, 可选) - 不为0时，用户神经网络所有层（bias和LayerNorm权重除外）的权重衰减。默认值： ``0`` 。
            - **num_epochs** (float, 可选) - 训练过程的总epoch数量。默认值： ``3.0`` 。
            - **gradient_accumulation_steps** (int, 可选) - 梯度累积中的间隔步数。默认值： ``1`` 。
            - **seed** (int, 可选) - 训练任务的随机数种子。默认值： ``42`` 。
            - **kwargs** (Any) - 其它参数。
