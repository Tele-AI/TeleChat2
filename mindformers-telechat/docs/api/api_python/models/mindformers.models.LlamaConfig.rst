mindformers.models.LlamaConfig
==============================

.. py:class:: mindformers.models.LlamaConfig(batch_size: int = 1,seq_length: int = 2048,hidden_size: int = 4096,num_layers: int = 32,num_heads: int = 32,n_kv_heads: Optional[int] = None,max_position_embedding: Optional[int] = None,intermediate_size: Optional[int] = None,vocab_size: int = 32000,multiple_of: int = 256,ffn_dim_multiplier: Optional[int] = None,rms_norm_eps: float = 1e-5,bos_token_id: int = 1,eos_token_id: int = 2,pad_token_id: int = 0,ignore_token_id: int = -100,theta: float = 10000.0,compute_dtype: str = "float16",layernorm_compute_type: str = "float32",softmax_compute_type: str = "float32",rotary_dtype: str = "float32",param_init_type: str = "float16",embedding_init_type=None,qkv_has_bias: bool = False,qkv_concat: bool = False,parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,moe_config: Union[dict, MoEConfig] = default_moe_config,use_past: bool = False,extend_method: str = "None",scaling_factor: float = 1.0,is_dynamic: bool = False,use_rope_slice: bool = False,use_flash_attention: bool = False,use_ring_attention: bool = False,use_attn_mask_compression: bool = False,parallel_optimizer: bool = False,fine_grain_interleave: int = 1,pp_interleave_num: int = 1,offset: int = 0,checkpoint_name_or_path: str = "",repetition_penalty: float = 1.0,max_decode_length: int = 1024,block_size: int = 16,num_blocks: int = 512,top_k: int = 5,top_p: float = 1.0,do_sample: bool = True,quant_config: dict = None,tie_word_embeddings: bool = False,llm_backend: str = "",fused_rms_norm: bool = True,**kwargs)

    Llama 配置类，定义了模型大小。

    参数：
        - **batch_size** (int, 可选) - 输入数据的批量大小，用于预测。默认值： ``1`` 。
        - **seq_length** (int, 可选) - 输入 ids 的序列长度，默认值： ``2048`` 。
        - **vocab_size** (int, 可选) - 模型的词汇表大小。默认值： ``32000`` 。
        - **hidden_size** (int, 可选) - 编码器层和池化层的维度。默认值： ``4096`` 。
        - **num_layers** (int, 可选) - Transformer 编码器中隐藏层的数量。默认值： ``32`` 。
        - **num_heads** (int, 可选) - Transformer 编码器中每个注意力层的注意力头数。默认值： ``32`` 。
        - **multiple_of** (int, 可选) - SwiGLU 隐藏层大小的倍数，默认值： ``256`` 。
        - **n_kv_heads** (int, 可选) - 多组头注意力头数，默认值： ``None`` 。
        - **ffn_dim_multiplier** (int, 可选) - ffn 层维度的倍数，默认值： ``None`` 。
        - **rms_norm_eps** (float, 可选) - rms_norm的epsilon 值。默认值： ``1e-5`` 。
        - **bos_token_id** (int, 可选) - *序列开始* 词元的 id。默认值： ``1`` 。
        - **eos_token_id** (int, 可选) - *序列结束* 词元的 id。默认值： ``2`` 。
        - **pad_token_id** (int, 可选) - *填充* 词元的 id。默认值： ``0`` 。
        - **ignore_token_id** (int, 可选) - *忽略* 词元的 id。默认值： ``-100`` 。
        - **compute_dtype** (str, 可选) - 线性层计算数据类型，默认值： ``float16`` 。
        - **layernorm_compute_type** (str, 可选) - layernorm 计算数据类型，默认值： ``float32`` 。
        - **softmax_compute_type** (str, 可选) - softmax 计算数据类型，默认值： ``float32`` 。
        - **rotary_dtype** (str, 可选) - rope 计算数据类型，默认值： ``float32`` 。
        - **param_init_type** (str, 可选) - 参数初始化数据类型，默认值： ``float16`` 。
        - **qkv_has_bias** (bool, 可选) - 查询、键和值的投影是否有偏置。默认值： ``False`` 。
        - **use_past** (bool, 可选) - 模型是否应使用过去的键/值注意力（如果适用于模型）来加速解码。默认值： ``False`` 。
        - **parallel_config** (TransformerOpParallelConfig) - 并行配置。默认值： ``default_transformer_config`` ，一个带有默认参数的 `TransformerOpParallelConfig` 实例。
        - **extend_method** (str, 可选) - 序列长度推理时的扩展方法。默认值： ``None`` 。
        - **use_flash_attention** (bool, 可选) - 是否启用闪存注意力操作。默认值： ``False`` 。
        - **use_ring_attention** (bool, 可选) - 是否启用环形注意力操作。默认值： ``False`` 。
        - **offset** (int, 可选) - 设置流水线阶段编号时的 Transformer 层偏移量。默认值： ``0`` 。
        - **checkpoint_name_or_path** (str, 可选) - 用于加载到网络中的检查点路径或名称。默认值： ``None`` 。
        - **repetition_penalty** (float, 可选) - 重复惩罚参数，1.0 表示没有惩罚。详细信息请参阅 `这篇论文 <。https://arxiv.org/pdf/1909.05858.pdf>`_ 。默认值： ``1.0`` 。
        - **max_decode_length** (int, 可选) - 生成的词元的最大长度，对应输入提示的长度加上 `max_new_tokens` 。如果同时设置了 `max_new_tokens` ，则它的效果将被覆盖。默认值： ``1024`` 。
        - **top_k** (int, 可选) - 用于 top-k 筛选的最高概率词汇表词元数量。默认值：``5`` 。
        - **top_p** (float, 可选) - 如果设置为小于 1 的浮点数，则仅保留概率和达到 `top_p` 或更高值的最小词元集合，用于生成。默认值： ``1.0`` 。
        - **do_sample** (bool, 可选) - 是否使用采样；否则使用贪婪解码。默认值： ``True`` 。
        - **block_size** (int, 可选) - 使用分页注意力时，一个块中可以包含的最大词元数。默认值： ``16`` 。
        - **num_blocks** (int, 可选) - 使用分页注意力时的最大块数。默认值： ``512`` 。
        - **tie_word_embeddings** (bool, 可选) - 是否将输入和输出嵌入层进行共享。默认值： ``False`` 。
        - **llm_backend** (str, 可选) - LLM 加速后端。默认值： ``None`` 。
        - **fused_rms_norm** (bool, 可选) - 是否使用融合算子的RMS_NORM。默认值： ``True`` 。

    返回：
        LlamaConfig 类实例。
