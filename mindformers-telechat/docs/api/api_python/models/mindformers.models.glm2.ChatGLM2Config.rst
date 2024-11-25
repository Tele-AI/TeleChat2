mindformers.models.glm2.ChatGLM2Config
=========================================================================

.. py:class:: mindformers.models.glm2.ChatGLM2Config(batch_size=1, num_layers=28, padded_vocab_size=65024, hidden_size=4096, ffn_hidden_size=13696, kv_channels=128, num_attention_heads=32, seq_length=2048, hidden_dropout=0.0, attention_dropout=0.0, layernorm_epsilon=1e-5, rope_ratio=1, rmsnorm=True, apply_residual_connection_post_layernorm=False, post_layer_norm=True, add_bias_linear=False, add_qkv_bias=True, bias_dropout_fusion=True, multi_query_attention=True, multi_query_group_num=2, apply_query_key_layer_scaling=True, attention_softmax_in_fp32=True, fp32_residual_connection=False, quantization_bit=0, pre_seq_len=None, prefix_projection=False, param_init_type: str = "float16", compute_dtype: str = "float16", layernorm_compute_type: str = "float32", rotary_dtype: str = None, use_past=False, use_flash_attention=False, block_size=16, num_blocks=128, is_dynamic=False, eos_token_id=2, pad_token_id=0, gmask_token_id=None, bos_token_id=None, repetition_penalty=1.0, checkpoint_name_or_path=None, parallel_config: Union[dict, TransformerOpParallelConfig]=default_transformer_config, offset=0, pp_interleave_num=1, **kwargs)

    ChatGLM2模型配置类，里面定义了模型的相关配置参数。

    参数：
        - **batch_size** (int，可选) - 输入数据的批量大小，用于推理。默认值： ``1`` 。
        - **num_layers** (int，可选) - Transformer编码器中隐藏层的数量。默认值： ``28`` 。
        - **padded_vocab_size** (int，可选) - ChatGLM2模型的词表大小。默认值： ``65024`` 。
        - **hidden_size** (int，可选) - 隐藏层的维度。默认值： ``4096`` 。
        - **ffn_hidden_size** (int，可选) - 前馈神经网络层的维度。默认值： ``13696`` 。
        - **kv_channels** (int，可选) - transformer中key和value向量的通道数。默认值： ``128`` 。
        - **num_attention_heads** (int，可选) - 每个注意力层的注意力头数量。默认值： ``32`` 。
        - **seq_length** (int，可选) - 输入语句对应token ids的序列长度。默认值： ``2048`` 。
        - **hidden_dropout** (float，可选) - dropout函数丢弃的比率值。默认值： ``0.0`` 。
        - **attention_dropout** (float，可选) - 注意力矩阵的dropout概率值。默认值： ``0.0`` 。
        - **layernorm_epsilon** (float，可选) - 计算层归一化时，防止分母为0而加上的ϵ值。默认值： ``1e-5`` 。
        - **rope_ratio** (float，可选) - RoPE旋转系数。默认值： ``1`` 。
        - **rmsnorm** (bool，可选) - 是否使用均方根归一化。默认值： ``True`` 。
        - **apply_residual_connection_post_layernorm** (bool，可选) - 残差连接层是否使用层归一化。默认值： ``False`` 。
        - **post_layer_norm** (bool，可选) - ffn层之后是否使用层归一化。默认值： ``True`` 。
        - **add_bias_linear** (bool，可选) - 线性层是否添加偏置。默认值： ``False`` 。
        - **add_qkv_bias** (bool，可选) - qkv是否添加偏置。默认值： ``True`` 。
        - **bias_dropout_fusion** (bool，可选) - 是否添加偏置、dropout、融合操作。默认值： ``True`` 。
        - **multi_query_attention** (bool，可选) - 是否使用mqa。默认值： ``True`` 。
        - **multi_query_group_num** (int，可选) - 定义多头注意力头的数量。默认值： ``2`` 。
        - **apply_query_key_layer_scaling** (bool，可选) - 是否对query和key进行缩放。默认值： ``True`` 。
        - **attention_softmax_in_fp32** (bool，可选) - 注意力层中的softmax层的计算类型是否使用float32。默认值： ``True`` 。
        - **fp32_residual_connection** (bool，可选) - 残差连接层中的计算类型是否使用float32。默认值： ``False`` 。
        - **quantization_bit** (int，可选) - 权重和激活的比特数。默认值： ``0`` 。
        - **pre_seq_len** (int，可选) - 输入序列前可学习的序列长度。默认值： ``None`` 。
        - **prefix_projection** (bool，可选) - 输入序列前是否添加投影层。默认值： ``False`` 。
        - **param_init_type** (str，可选) - 参数初始化时的数据类型。默认值： ``float16`` 。
        - **compute_dtype** (str，可选) - 线性层数据计算类型。默认值： ``float16`` 。
        - **layernorm_compute_type** (str，可选) - 归一化层的计算类型。默认值： ``float32`` 。
        - **use_past** (bool，可选) - 是否采用增量推理。默认值： ``False`` 。
        - **use_flash_attention** (bool，可选) - 是否使用flash attention算法。默认值： ``False`` 。
        - **block_size** (int，可选) - 使用PagedAttention时，一个分块中可以有的最大token数。默认值： ``16`` 。
        - **num_blocks** (int，可选) - 使用PagedAttention时的最大块数。默认值： ``128`` 。
        - **is_dynamic** (bool，可选) - 是否使用动态图模式。默认值： ``False`` 。
        - **eos_token_id** (int，可选) - 推理结束标志的token id值。默认值： ``2`` 。
        - **pad_token_id** (int，可选) - 在多batch推理时，对较短序列进行填充补齐的token id值。默认值： ``0`` 。
        - **gmask_token_id** (int，可选) - 一个特殊的token id，该标记用于指示模型在自注意力机制中对某些位置的标记进行全局关注。默认值： ``None`` 。
        - **bos_token_id** (int，可选) - 标记序列第一个元素的token id值，用于指示序列的开始。默认值： ``None`` 。
        - **repetition_penalty** (float，可选) - 重复惩罚的参数。默认值： ``1.0`` 。
        - **checkpoint_name_or_path** (str，可选) - 模型名称或本地加载路径。默认值： ``None`` 。
        - **parallel_config** (TransformerOpParallelConfig，可选) - 模型并行化处理的参数配置。默认值： ``default_transformer_config`` 。
        - **offset** (int，可选) - 每个（微批量）阶段的层偏移。默认值： ``0`` 。
        - **pp_interleave_num** (int，可选) - 流水线并行中微批次交织的次数。默认值： ``1`` 。
        - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。
