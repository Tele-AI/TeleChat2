mindformers.Trainer
====================

.. py:class:: mindformers.Trainer(args: Optional[Union[str, MindFormerConfig, TrainingArguments]] = None, task: Optional[str] = 'general', model: Optional[Union[str, PreTrainedModel]] = None, model_name: Optional[str] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None, train_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None, eval_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None, data_collator: Optional[Callable] = None, optimizers: Optional[Optimizer] = None, compute_metrics: Optional[Union[dict, set]] = None, callbacks: Optional[Union[Callback, List[Callback]]] = None, eval_callbacks: Optional[Union[Callback, List[Callback]]] = None, pet_method: Optional[str] = '', image_processor: Optional[BaseImageProcessor] = None, audio_processor: Optional[BaseAudioProcessor] = None, save_config: bool = False, reset_model: bool = False)

    Trainer是通用的任务执行组件，通过参数中的task名称或者配置文件可以指定任务并且初始化一个与该任务相关的trainer实例。用户可以通过封装trainer实例中的train, finetune, evaluate和predict来实现不同的任务。同时，用户还可以自定义model, optimizer, dataset, tokenizer, processor, train_one_step, callback, metric等。

    可以通过以下方式对Trainer进行初始化：

    1. 定义 `task` 和 `model_name`，例如：task='text_generation'，model_name='gpt2'。指定正确的 `task` 和 `model_name`，便可以通过MindFormerBook发现相应的YAML配置，并且将YAML信息任务的配置。
    2. 定义 `task` 和 `model`，例如：task='text_generation'，model='gpt2'。其中， `model` 可以是一个模型实例或者模型名称，如果 `model` 是模型名称，它会覆盖 `model_name`。
    3. 定义 `task`、 `model_name` 和 `model`，此时 `model` 是模型实例。
    4. 定义 `args` 为MindFormerConfig实例或者YAML文件路径。此时也可以通过 `model` 参数传递模型实例，否则会通过 `args` 配置信息来实例化模型。
    5. 定义 `args` 为TrainingArguments实例，并且 `model` 是模型实例。
    6. 定义 `args` 为TrainingArguments实例，并且定义 `task` 和 `model_name`。此时不需要传递模型实例，先通过 `task` 和 `model_name` 获得YAML配置，然后再通过YAML配置来实例化模型。

    .. note::
        1. 如果同时传递了 `args` 、 `task` 和 `model_name`，任务配置的优先级高于 `args`，通过 `task` 和 `model_name` 获得的YAML配置会被 `args` 覆盖。
        2. 如果要使用Trainer.predict来进行推理，则 `task` 是必需的。

    参数：
        - **args** (Union[str, MindFormerConfig, TrainingArguments], 可选) - 任务的配置，用于初始化数据集、超参、优化器等。支持yaml文件路径、MindFormerConfig类或者TrainingArguments类等。默认值： ``None`` 。
        - **task** (str, 可选) - 任务类型。默认值： ``general`` 。
        - **model** (Union[str, PreTrainedModel], 可选) - 训练的神经网络，模型名称或者PreTrainedModel实例。默认值： ``None`` 。
        - **model_name** (str, 可选) - 模型名称。默认值： ``None`` 。
        - **tokenizer** (PreTrainedTokenizerBase, 可选) - 文本处理的分词器，支持PreTrainedTokenizerBase类。默认值： ``None`` 。
        - **train_dataset** (Union[str, BaseDataset, Dataset, Iterable], 可选) - 训练数据集，支持真实数据集路径或者BaseDateset类，或者MindSpore的Dataset类。默认值： ``None`` 。
        - **eval_dataset** (Union[str, BaseDataset, Dataset, Iterable], 可选) - 评估数据集，支持真实数据集路径或者BaseDateset类，或者MindSpore的Dataset类。默认值： ``None`` 。
        - **data_collator** (Callable, 可选) - 批量数据处理的方法。默认值： ``None`` 。
        - **optimizers** (Optimizer, 可选) - 训练神经网络的优化器，支持MindSpore的Optimizer类。默认值： ``None`` 。
        - **compute_metrics** (Union[dict, set], 可选) - 评估指标，在MindSpore的Metric类中支持dict或set类型。默认值： ``None`` 。
        - **callbacks** (Union[Callback, List[Callback]], 可选) - 训练的回调函数，支持MindSpore中的CallBack或者CallBack列表。默认值： ``None`` 。
        - **eval_callbacks** (Union[Callback, List[Callback]], 可选) - 评估的回调，支持MindSpore中的CallBack或者CallBack列表。默认值： ``None`` 。
        - **pet_method** (str, 可选) - 参数高效微调(Pet: Parameter-Efficient Tuning)方法。默认值： ``''`` 。
        - **image_processor** (BaseImageProcessor, 可选) - 图像预处理的处理器，支持BaseImageProcessor类。默认值： ``None`` 。
        - **audio_processor** (BaseAudioProcessor, 可选) - 音频预处理的处理器，支持BaseAudioProcessor类。默认值： ``None`` 。
        - **save_config** (bool, 可选) - 保存任务当前的配置。默认值： ``False`` 。
        - **reset_model** (bool, 可选) - 重置模型实例。默认值： ``False`` 。

    返回：
        Trainer类的实例。

    异常：
        - **KeyError** - `task` 或者 `model` 参数不支持。

    .. py:method:: evaluate(eval_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None, eval_checkpoint: Optional[Union[str, bool]] = False, auto_trans_ckpt: Optional[bool] = None, src_strategy: Optional[str] = None, transform_process_num: Optional[int] = None, **kwargs)

        Trainer中执行评估的API，在设置了用户自定义的配置后，通过调用Trainer实例的evaluate方法来执行评估。

        参数：
            - **eval_dataset** (Union[str, BaseDataset, Dataset, Iterable], 可选) - 评估数据集。默认值： ``None`` 。
            - **eval_checkpoint** (Union[str, bool], 可选) - 用于设置神经网络的权重。支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。默认值： ``False`` 。
            - **auto_trans_ckpt** (bool, 可选) - 自动转换checkpoint，加载到分布式的模型中。默认值： ``None`` 。
            - **src_strategy** (str, 可选) - 加载checkpoint的策略，只有auto_trans_ckpt为True时才生效。默认值： ``None`` 。
            - **transform_process_num** (int, 可选) - 转换checkpoint的进程数。默认值： ``None`` 。
            - **kwargs** (Any) - 其它参数。

        异常：
            - **TypeError** - `eval_checkpoint` 不是bool或者str类型。

    .. py:method:: finetune(finetune_checkpoint: Optional[Union[str, bool]] = False, resume_from_checkpoint: Optional[Union[str, bool]] = None, resume_training: Optional[Union[bool, str]] = None, ignore_data_skip: Optional[bool] = None, data_skip_steps: Optional[int] = None, auto_trans_ckpt: Optional[bool] = None, src_strategy: Optional[str] = None, transform_process_num: Optional[int] = None, do_eval: bool = False)

        Trainer中执行微调的API，在设置了用户自定义的配置后，通过调用Trainer实例的finetune方法来执行微调。

        参数：
            - **finetune_checkpoint** (Union[str, bool], 可选) - 在训练或者微调中，用于重新设置神经网络的权重，支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。默认值： ``False`` 。
            - **resume_from_checkpoint** (Union[str, bool], 可选) - 在训练或者微调中，用于重新设置神经网络的权重，支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。如果finetune_checkpoint有传入的话，resume_from_checkpoint会被覆盖。默认值： ``None`` 。
            - **resume_training** (Union[bool, str], 可选) - 指定是否恢复训练，或者指定checkpoint名称来恢复训练。如果值为True，则加载meta.json中指定的checkpoint来恢复训练。如果指定的是checkpoint名称，则该名称的checkpoint会被加载用于恢复训练。默认值： ``None`` 。
            - **ignore_data_skip** (bool, 可选) - 在恢复训练时，是否跳过执行过的批次，加载与前一次训练相同阶段的数据。如果值为True，则训练任务启动更快（因为跳过了一些步骤），但是由于训练被中断，所以无法获得相同的结果。默认值： ``None`` 。
            - **data_skip_steps** (int, 可选) - 在恢复训练时，指定在训练数据集中跳过的步数。只有 `ignore_data_skip` 值为False时生效。默认值： ``None`` 。
            - **auto_trans_ckpt** (bool, 可选) - 自动转换checkpoint，加载到分布式的模型中。默认值： ``None`` 。
            - **src_strategy** (str, 可选) - 加载checkpoint的策略，只有auto_trans_ckpt为True时才生效。默认值： ``None`` 。
            - **transform_process_num** (int, 可选) - 转换checkpoint的进程数。默认值： ``None`` 。
            - **do_eval** (bool, 可选) - 在训练中是否执行评估。默认值： ``False`` 。

        异常：
            - **TypeError** - `load_checkpoint` 不是bool或者str类型。

    .. py:method:: predict(predict_checkpoint: Optional[Union[str, bool]] = None, auto_trans_ckpt: Optional[bool] = None, src_strategy: Optional[str] = None, transform_process_num: Optional[int] = None, input_data: Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]] = None, batch_size: int = None, **kwargs)

        Trainer中执行预测的API，在设置了用户自定义的配置后，通过调用Trainer实例的predict方法来执行预测。

        参数：
            - **predict_checkpoint** (Union[str, bool], 可选) - 用于设置神经网络的权重。支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。默认值： ``None`` 。
            - **auto_trans_ckpt** (bool, 可选) - 自动转换checkpoint，加载到分布式的模型中。默认值： ``None`` 。
            - **src_strategy** (str, 可选) - 加载checkpoint的策略，只有auto_trans_ckpt为True时才生效。默认值： ``None`` 。
            - **transform_process_num** (int, 可选) - 转换checkpoint的进程数。默认值： ``None`` 。
            - **input_data** (Union[Tensor, np.ndarray, Image, str, list], 可选) - 输入数据。默认值： ``None`` 。
            - **batch_size** (int, 可选) - 输入数据的batch size。默认值： ``None`` 。
            - **kwargs** (Any) - 其它参数。

        返回：
            预测结果。

        异常：
            - **TypeError** - `predict_checkpoint` 不是bool或者str类型。
            - **TypeError** - `input_data` 不是Tensor、np.ndarray、Image、str或者list类型。

    .. py:method:: train(train_checkpoint: Optional[Union[str, bool]] = False, resume_from_checkpoint: Optional[Union[str, bool]] = None, resume_training: Optional[Union[bool, str]] = None, ignore_data_skip: Optional[bool] = None, data_skip_steps: Optional[int] = None, auto_trans_ckpt: Optional[bool] = None, src_strategy: Optional[str] = None, transform_process_num: Optional[int] = None, do_eval: Optional[bool] = False)

        Trainer中执行训练的API，在设置了用户自定义的配置后，通过调用Trainer实例的train方法来执行训练。

        参数：
            - **train_checkpoint** (Union[str, bool], 可选) - 在训练或者微调中，用于重新设置神经网络的权重，支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。默认值： ``False`` 。
            - **resume_from_checkpoint** (Union[str, bool], 可选) - 在训练或者微调中，用于重新设置神经网络的权重，支持真实的checkpoint路径、MindFormers中的模型名称，或者布尔值。如果值为True，则自动使用上一轮训练保存的checkpoint文件。如果train_checkpoint有传入的话，resume_from_checkpoint会被覆盖。默认值： ``None`` 。
            - **resume_training** (Union[bool, str], 可选) - 指定是否恢复训练，或者指定checkpoint名称来恢复训练。如果值为True，则加载meta.json中指定的checkpoint来恢复训练。如果指定的是checkpoint名称，则该名称的checkpoint会被加载用于恢复训练。默认值： ``None`` 。
            - **ignore_data_skip** (bool, 可选) - 在恢复训练时，是否跳过执行过的批次，加载与前一次训练相同阶段的数据。如果值为True，则训练任务启动更快（因为跳过了一些步骤），但是由于训练被中断，所以无法获得相同的结果。默认值： ``None`` 。
            - **data_skip_steps** (int, 可选) - 在恢复训练时，指定在训练数据集中跳过的步数。只有 `ignore_data_skip` 值为False时生效。默认值： ``None`` 。
            - **auto_trans_ckpt** (bool, 可选) - 自动转换checkpoint，加载到分布式的模型中。默认值： ``None`` 。
            - **src_strategy** (str, 可选) - 加载checkpoint的策略，只有auto_trans_ckpt为True时才生效。默认值： ``None`` 。
            - **transform_process_num** (int, 可选) - 转换checkpoint的进程数。默认值： ``None`` 。
            - **do_eval** (bool, 可选) - 在训练中是否执行评估。默认值： ``False`` 。

        异常：
            - **TypeError** - `resume_from_checkpoint` 不是bool或者str类型。
