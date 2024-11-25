mindformers.generation.GenerationMixin
======================================

.. py:class:: mindformers.generation.GenerationMixin

    一个提供自回归文本生成的所有函数的类，作为PreTrainedModel的织入（mixin）使用。

    .. py:method:: chat(tokenizer: PreTrainedTokenizer, query: str, history: Optional[List[Dict[str, str]]] = None, system_role_name: Optional[str] = "system", user_role_name: Optional[str] = "user", assistant_role_name: Optional[str] = "assistant", instruction: Optional[str] = "", max_length: Optional[int] = 512, max_new_tokens: Optional[int] = None, min_length: Optional[int] = 0, min_new_tokens: Optional[int] = None, do_sample: Optional[bool] = True, temperature: Optional[float] = 1.0, top_k: Optional[int] = 50, top_p: Optional[float] = 1.0, repetition_penalty: Optional[float] = 1.0)

        基于大型语言模型的对话文本生成推理。通过提供的分词器添加聊天模板后，将使用 `generate()` 对用户的查询进行推断。

        参数：
            - **tokenizer** (PreTrainedTokenizer) - 用于解码词元的分词器。
            - **query** (str) - 用户对于推理的输入。
            - **history** (List[Dict[str, str]], 可选) - 一个对话对象，或包含有 ``"role"`` 和 ``"content"`` 键的字典列表，代表到目前为止的聊天记录。默认值： ``None`` 。
            - **system_role_name** (str) - 系统角色的名称。默认值： ``"system"`` 。
            - **user_role_name** (str) - 用户角色的名称。默认值： ``"user"`` 。
            - **assistant_role_name** (str) - 助理角色的名称。默认值： ``"assistant"`` 。
            - **instruction** (str, 可选) - 给模型的指令消息。默认值： ``""`` 。
            - **max_length** (int, 可选) - 生成词元的最大长度。对应于输入提示符的长度+ `max_new_tokens` 。如果之前设置过 `max_new_tokens` ，那么现在将会覆盖其效果。默认值： ``512`` 。
            - **max_new_tokens** (int, 可选) - 要生成的最大词元数，忽略提示符中的词元数。默认值： ``None`` 。
            - **min_length** (int, 可选) - 要生成的序列的最小长度。对应于输入提示符的长度+ `min_new_tokens` 。如果之前设置过 `min_new_tokens` ，那么现在将会覆盖其效果。默认值： ``0`` 。
            - **min_new_tokens** (int, 可选) - 要生成的最小标记数，忽略提示符中的标记数。默认值： ``None`` 。
            - **do_sample** (bool, 可选) - 是否对候选索引进行采样。如果将其设置为 ``True`` ，则启用抽样；如果设置为 ``False`` ，则禁用抽样，相当于top-k 1。如果将其设置为 ``None`` ，则遵循模型配置中的设置。默认值： ``True`` 。
            - **temperature** (float, 可选) - 用于调制下一个词元概率的值。默认值： ``1.0`` 。
            - **top_k** (int, 可选) - 确定top-k数字词元索引作为候选。这个参数取值应该为正数。如果将其设置为 ``None`` ，则遵循模型配置中的设置。默认值： ``50`` 。
            - **top_p** (float, 可选) - top-p下面的候选词元索引的累积概率，将被选择为候选索引。top-p的有效值在(0,1]之间。如果该值大于1，表示启用top-k算法。如果将其设置为 ``None`` ，则遵循模型配置中的设置。默认值： ``1.0`` 。
            - **repetition_penalty** (float, 可选) - 生成单词频率的惩罚因子。如果将其设置为1，则不启用 `repeat_penalty` 。如果将其设置为 ``None`` ，则遵循模型配置中的设置。其默认值： ``1.0`` 。

        返回：
            两个参数， `response` 表示本次会话中大模型的回复结果，`history` 表示对话历史。

    .. py:method:: forward(input_ids: [Union[List[int], List[List[int]]]], valid_length_each_example: np.ndarray, block_tables: Optional[Tensor] = None, slot_mapping: Optional[Tensor] = None, prefill: bool = None, use_past: bool = False, encoder_mask: Optional[Tensor] = None, encoder_output: Optional[Tensor] = None, target_mask: Optional[Tensor] = None, **model_kwargs)

        模型前向传播的过程。

        参数：
            - **input_ids** (List(List(int))) - 填充（Padding）后的输入索引。
            - **valid_length_each_example** (np.ndarray) - 除填充外的有效输入长度。
            - **block_tables** (Tensor) - 页面注意力的参数。
            - **slot_mapping** (Tensor) - 页面注意力的参数。
            - **prefill** (bool) - 选择是进行预填充预测还是解码预测。
            - **use_past** (bool) - 选择是否使用过去的状态。
            - **encoder_mask** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。
            - **encoder_output** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。
            - **target_mask** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。

        返回：
            两个参数，`res` 返回前向传播处理后的结果，`current_index` 记录序列的当前索引。

    .. py:method:: generate(input_ids: Optional[Union[List[int], List[List[int]]]], generation_config: Optional[GenerationConfig] = None, logits_processor: Optional[LogitsProcessorList] = None, streamer: Optional[BaseStreamer] = None, seed: Optional[int] = None, **kwargs)

        可以根据给定的input ids（即数字id数组，本质是词元索引）来生成词汇。

        大多数生成控制参数都在 `generation_config` 中进行配置，如果它们没有被传递，则将应用模型的默认生成配置。你可以通过传递相应的参数给 `generate()` 来随意重写 `generation_config` ，比如 ``.generate(inputs, top_k=3, do_sample=True)`` 。

        参数：
            - **input_ids** (List(str), List(List(str))) - 单个词元索引列表或一批词元索引列表。当输入为一批词元索引列表时，要求每个词元索引列表的长度保持一致。
            - **generation_config** (`GenerationConfig`, 可选) - 用生成配置来对生成调用进行基本参数化。 ``**kwargs`` 作为参数列表，会传递到与 `generation_config` 相匹配的属性处，并将覆盖默认值。如果没有提供 `generation_config` ，则将使用到模型配置中的默认配置。请注意，未指定的参数将继承[`GenerationConfig`]的默认值，应该检查其文档以进行参数化。默认值： ``None`` 。
            - **logits_processor** (`LogitsProcessorList`, 可选) - 自定义置信度处理器，补充了由参数和生成配置构建的默认置信度处理器。如果传递了一个已经用参数或生成配置创建的置信度处理器，则会抛出错误。本特性适用于高级用户。默认值： ``None`` 。
            - **streamer** (TextStreamer) - 生成器使用的streamer。
            - **seed** (int) - 样本中使用的随机种子。

            - **kwargs** - `generate_config` 的特定参数化和（或）其他特定于模型的kwargs，这些kwargs将被转发给模型的 `forward` 函数。受支持的 `generate_config` 关键字可以在[`GenerationConfig`]的文档中检查。主要使用到的关键词如下:

              - **max_length** (int) - 生成词元的最大长度。对应于输入提示符的长度+ `max_new_tokens` 。如果之前设置了 `max_new_tokens` ，则将其效果覆盖。
              - **max_new_tokens** (int) - 要生成的最大词元数，忽略提示符中的词元数。
              - **min_length** (int) - 要生成的序列的最小长度。对应于输入提示符的长度+ `min_new_tokens` 。如果之前设置了 `min_new_tokens` ，则将其效果覆盖。
              - **min_new_tokens** (int) - 要生成的最小词元数，忽略提示符中的词元数。
              - **do_sample** (bool) - 是否对候选索引进行采样。如果将其设置为 ``True`` ，则启用抽样；如果设置为 ``False``，则禁用抽样，相当于top-k 1。如果将其设置为 ``None`` ，则遵循模型配置中的设置。
              - **top_k** (int) - 确定top-k数字词元索引作为候选。这个参数取值应该为正数。如果将其设置为 ``None``，则遵循模型配置中的设置。
              - **top_p** (float) - top-p下面的候选词元索引的累积概率，将被选择为候选索引。top-p的有效值在(0,1]之间。如果该值大于1，表示启用top-k算法。如果将其设置为 ``None`` ，则遵循模型配置中的设置。
              - **eos_token_id** (int) - 句子结束的词元索引。如果设置为None，则遵循模型配置中的设置。
              - **pad_token_id** (int) - 填充的词元索引。如果设置为None，则遵循模型配置中的设置。
              - **repetition_penalty** (float) - 生成单词频率的惩罚因子。如果将其设置为1，则不启用 `repeat_penalty` 。如果将其设置为 ``None`` ，则遵循模型配置中的设置。默认值： ``None`` 。
              - **num_beams** (int) - 用于束搜寻的束的数量。1表示不使用束搜寻。如果大于1，则 `do_sample` 将被设置为 ``False`` 。

        返回：
            生成的一个词元索引列表。

    .. py:method:: infer(input_ids: Union[List[int], List[List[int]]], valid_length_each_example: np.ndarray, generation_config: GenerationConfig = None, logits_processor: Optional[LogitsProcessorList] = None, logits_warper: Optional[LogitsProcessorList] = None, block_tables: Optional[Tensor] = None, slot_mapping: Optional[Tensor] = None, prefill: bool = True, is_finished: List[bool] = None, encoder_mask: Optional[Tensor] = None, encoder_output: Optional[Tensor] = None, target_mask: Optional[Tensor] = None, **model_kwargs)

        用于对下一个位置做推断并返回其置信度，可以选择来用做预填充或解码预测。

        参数：
            - **input_ids** (List(List(int))) - 填充（Padding）后的输入索引。
            - **valid_length_each_example** (np.ndarray) - 除填充外的有效输入长度。
            - **generation_config** (`GenerationConfig`) - 用生成配置来对生成调用进行基本参数化。
            - **logits_processor** (`LogitsProcessorList`, 可选) - [`LogitsProcessorList`]的一个实例。这是由继承自[`LogitsProcessor`]类的实例组成的一个列表，用于在每一步生成过程中修改语言模型头部的预测得分。默认值： ``None`` 。
            - **logits_warper** (`LogitsProcessorList`, 可选) - [`LogitsProcessorList`]的一个实例。这是一个由继承自[`LogitsWarper`]类的实例组成的列表，用于在每一步生成过程中的多项式采样之前，调整语言模型头部的预测得分分布。默认值： ``None`` 。
            - **block_tables** (Tensor) - 页面注意力的参数。
            - **slot_mapping** (Tensor) - 页面注意力的参数。
            - **prefill** (bool) - 选择是进行预填充预测还是解码预测。
            - **is_finished** (List(bool)) - 选择每个序列是否完成了生成。
            - **encoder_mask** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。
            - **encoder_output** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。
            - **target_mask** (Tensor) - 用于编码器-解码器结构，对于纯解码器结构则不需要。

        返回：
            两个参数，`next_token` 表示生成的下一个词元，`is_finished` 表示当前批次是否完成了序列生成任务。

    .. py:method:: postprocess(input_ids, is_finished, res, generation_config: GenerationConfig, valid_length_each_example, current_index: Optional[Union[List[int], List[List[int]]]], logits_processor: Optional[LogitsProcessorList] = None, logits_warper: Optional[LogitsProcessorList] = None, need_gather_logits: bool = True)

        模型生成输出的后处理。

        参数：
            - **input_ids** (List(List(int))) - 填充（Padding）后的输入索引。
            - **res** (List(List(int))) - 推断后的置信度。
            - **is_finished** (List(bool)) - 记录每个序列是否完成其生成。
            - **generation_config** (`GenerationConfig`) - 生成配置用作生成调用的基本参数化。
            - **valid_length_each_example** (np.ndarray) - 除填充外的有效输入长度。
            - **current_index** (List(int)) - 序列的当前索引。
            - **logits_processor** (`LogitsProcessorList`, 可选) - [`LogitsProcessorList`]的一个实例。这是由继承自[`LogitsProcessor`]类的实例组成的一个列表，用于在每一步生成过程中修改语言模型头部的预测得分。默认值： ``None`` 。
            - **logits_warper** (`LogitsProcessorList`, 可选) - [`LogitsProcessorList`]的一个实例。这是一个由继承自[`LogitsWarper`]类的实例组成的列表，用于在每一步生成过程中的多项式采样之前，调整语言模型头部的预测得分分布。默认值： ``None`` 。
            - **need_gather_logits** (bool) - 在解码预测且为第一次迭代时是否收集结果，设置为True。

        返回：
            四个参数，`target_list` 表示本次处理的目标列表，`next_probs_cache` 和 `next_logits_cache` 分别用作存储置信度和文本输出概率的缓存，`is_finished` 表示当前批次是否完成了序列生成任务。