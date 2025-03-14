mindformers.models.LlamaConfig
==============================

.. py:class:: mindformers.models.LlamaConfig(batch_size: int = 1,seq_length: int = 2048,hidden_size: int = 4096,num_layers: int = 32,num_heads: int = 32,n_kv_heads: Optional[int] = None,max_position_embedding: Optional[int] = None,intermediate_size: Optional[int] = None,vocab_size: int = 32000,multiple_of: int = 256,ffn_dim_multiplier: Optional[int] = None,rms_norm_eps: float = 1e-5,bos_token_id: int = 1,eos_token_id: int = 2,pad_token_id: int = 0,ignore_token_id: int = -100,theta: float = 10000.0,compute_dtype: str = "float16",layernorm_compute_type: str = "float32",softmax_compute_type: str = "float32",rotary_dtype: str = "float32",param_init_type: str = "float16",residual_dtype: str = None,embedding_init_type=None,qkv_has_bias: bool = False,qkv_concat: bool = False,attn_proj_has_bias: bool = False,parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,moe_config: Union[dict, MoEConfig] = default_moe_config,use_past: bool = False,extend_method: str = "None",scaling_factor: float = 1.0,is_dynamic: bool = False,use_rope_slice: bool = False,use_flash_attention: bool = False,use_ring_attention: bool = False,use_attn_mask_compression: bool = False,parallel_optimizer: bool = False,fine_grain_interleave: int = 1,pp_interleave_num: int = 1,offset: int = 0,init_method_std: float = 0.01,checkpoint_name_or_path: str = "",repetition_penalty: float = 1.0,max_decode_length: int = 1024,block_size: int = 16,num_blocks: int = 512,top_k: int = 5,top_p: float = 1.0,do_sample: bool = True,quant_config: dict = None,tie_word_embeddings: bool = False,llm_backend: str = "",fused_rms_norm: bool = True,input_sliced_sig: bool = False,rmsnorm_compute_2d: bool = False,chunk_prefill: bool = False,calculate_per_token_loss: bool = False,pipeline_stage: dict = None,**kwargs)

    Llama 配置类，定义了模型大小。

    参数：
        - **batch_size** (int, 可选) - 输入数据的批量大小，用于预测。默认值： ``1`` 。
        - **seq_length** (int, 可选) - 输入 ids 的序列长度。默认值： ``2048`` 。
        - **hidden_size** (int, 可选) - 编码器层和池化层的维度。默认值： ``4096`` 。
        - **num_layers** (int, 可选) - Transformer 编码器中隐藏层的数量。默认值： ``32`` 。
        - **num_heads** (int, 可选) - Transformer 编码器中每个注意力层的注意力头数。默认值： ``32`` 。
        - **n_kv_heads** (int, 可选) - 多组头注意力头数。默认值： ``None`` 。
        - **max_position_embedding** (int, 可选) - 自定义模型可处理的最大序列长度。默认值： ``None`` 。
        - **intermediate_size** (int, 可选) - 自定义中间层的维数。默认值： ``None`` 。
        - **vocab_size** (int, 可选) - 模型的词汇表大小。默认值： ``32000`` 。
        - **multiple_of** (int, 可选) - SwiGLU 隐藏层大小的倍数。默认值： ``256`` 。
        - **ffn_dim_multiplier** (int, 可选) - ffn 层维度的倍数。默认值： ``None`` 。
        - **rms_norm_eps** (float, 可选) - rms_norm的epsilon 值。默认值： ``1e-5`` 。
        - **bos_token_id** (int, 可选) - *序列开始* 词元的 id。默认值： ``1`` 。
        - **eos_token_id** (int, 可选) - *序列结束* 词元的 id。默认值： ``2`` 。
        - **pad_token_id** (int, 可选) - *填充* 词元的 id。默认值： ``0`` 。
        - **ignore_token_id** (int, 可选) - *忽略* 词元的 id。默认值： ``-100`` 。
        - **theta** (float, 可选) - RoPE 中正弦和余弦函数的频率因子。默认值： ``10000.0`` 。
        - **compute_dtype** (str, 可选) - 线性层计算数据类型，默认值： ``float16`` 。
        - **layernorm_compute_type** (str, 可选) - layernorm 计算数据类型，默认值： ``float32`` 。
        - **softmax_compute_type** (str, 可选) - softmax 计算数据类型，默认值： ``float32`` 。
        - **rotary_dtype** (str, 可选) - rope 计算数据类型，默认值： ``float32`` 。
        - **param_init_type** (str, 可选) - 参数初始化数据类型，默认值： ``float16`` 。
        - **residual_dtype** (str, 可选) - 残差计算数据类型，默认值： ``None``。
        - **embedding_init_type** (str, 可选) - 嵌入权重初始化数据类型。默认值： ``None`` 。
        - **qkv_has_bias** (bool, 可选) - Query,、Key和Value的投影是否有偏置。默认值： ``False`` 。
        - **qkv_concat** (bool, 可选) - 是否拼接Query,、Key和Value投影。默认值： ``None`` 。
        - **attn_proj_has_bias** (bool, 可选) - 注意力中的投影是否有偏置。默认值： ``False``。
        - **parallel_config** (Union[dict, TransformerOpParallelConfig], 可选) - 并行配置。默认值： ``default_transformer_config`` ，一个带有默认参数的 `TransformerOpParallelConfig` 实例。
        - **moe_config** (Union[dict, MoEConfig], 可选) - MoE配置。默认值： ``default_moe_config`` ，一个带有默认参数的 `MoEConfig` 实例。
        - **use_past** (bool, 可选) - 模型是否应使用过去的Key和Value注意力（如果适用于模型）来加速解码。默认值： ``False`` 。
        - **extend_method** (str, 可选) - 序列长度推理时的扩展方法。默认值： ``None`` 。
        - **scaling_factor** (float, 可选) - 缩放因子，用于调整正弦和余弦函数中频率因子的权重。默认值： ``1.0`` 。
        - **is_dynamic** (bool, 可选) - 是否使用动态shape。默认值： ``False`` 。
        - **use_rope_slice** (bool, 可选) - 是否启用 RoPE 切分。默认值： ``False`` 。
        - **use_flash_attention** (bool, 可选) - 是否启用闪存注意力操作。默认值： ``False`` 。
        - **use_ring_attention** (bool, 可选) - 是否启用环形注意力操作。默认值： ``False`` 。
        - **use_attn_mask_compression** (bool, 可选) - 是否启用注意力掩码压缩。默认值： ``False`` 。
        - **parallel_optimizer** (bool, 可选) - 是否启用优化器并行。默认值： ``False`` 。
        - **fine_grain_interleave** (int, 可选) - 设置细粒度多副本的数量。默认值： ``1`` 。
        - **pp_interleave_num** (int, 可选) - 设置交织流水的数量。默认值： ``1`` 。
        - **offset** (int, 可选) - 设置流水线阶段编号时的 Transformer 层偏移量。默认值： ``0`` 。
        - **init_method_std** (float, 可选) - 线性映射层使用normal类型初始化的标准差。默认值： ``0.01``。
        - **checkpoint_name_or_path** (str, 可选) - 用于加载到网络中的检查点路径或名称。默认值： ``None`` 。
        - **repetition_penalty** (float, 可选) - 重复惩罚参数，1.0 表示没有惩罚。详细信息请参阅 `这篇论文 <。https://arxiv.org/pdf/1909.05858.pdf>`_ 。默认值： ``1.0`` 。
        - **max_decode_length** (int, 可选) - 生成的词元的最大长度，对应输入提示的长度加上 `max_new_tokens` 。如果同时设置了 `max_new_tokens` ，则它的效果将被覆盖。默认值： ``1024`` 。
        - **block_size** (int, 可选) - 使用分页注意力时，一个块中可以包含的最大词元数。默认值： ``16`` 。
        - **num_blocks** (int, 可选) - 使用分页注意力时的最大块数。默认值： ``512`` 。
        - **top_k** (int, 可选) - 用于 top-k 筛选的最高概率词汇表词元数量。默认值：``5`` 。
        - **top_p** (float, 可选) - 如果设置为小于 1 的浮点数，则仅保留概率和达到 `top_p` 或更高值的最小词元集合，用于生成。默认值： ``1.0`` 。
        - **do_sample** (bool, 可选) - 是否使用采样；否则使用贪婪解码。默认值： ``True`` 。
        - **quant_config** (dict, 可选) - 量化配置。默认值： ``None`` 。
        - **tie_word_embeddings** (bool, 可选) - 是否将输入和输出嵌入层进行共享。默认值： ``False`` 。
        - **llm_backend** (str, 可选) - LLM 加速后端。默认值： ``None`` 。
        - **fused_rms_norm** (bool, 可选) - 是否使用融合算子的RMS_NORM。默认值： ``True`` 。
        - **input_sliced_sig** (bool, 可选) - 数据集是否已处理成模型的seq_length大小。默认值：``False``。
        - **rmsnorm_compute_2d** (bool, 可选) - RMS_NORM中的加算子是否转2维实现。默认值：``False``。
        - **chunk_prefill** (bool, 可选) - 是否开启全量和增量混合的推理。默认值： ``False`` 。
        - **calculate_per_token_loss** (bool, 可选) - 是否计算每个token的损失。默认值： ``False`` 。
        - **pipeline_stage** (dict, 可选) - 一个设置流水并行时模型的start_stage、stage_num和offset。默认值： ``None`` 。

    返回：
        LlamaConfig 类实例。
