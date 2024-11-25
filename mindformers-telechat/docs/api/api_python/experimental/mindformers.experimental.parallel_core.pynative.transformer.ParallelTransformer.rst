mindformers.experimental.parallel_core.pynative.transformer.ParallelTransformer
===============================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.transformer.ParallelTransformer(config, model_type, layer_type=1, self_attn_mask_type=1, post_norm=True, pre_process=False, post_process=False, drop_path_rate=0.0)

    这个类代表一个并行的transformer层。它由多个transformer层组成，可以处理各种配置和处理步骤。

    参数：
        - **config** (dict) - 一个配置字典，提供了并行transformer层的各种参数配置。
        - **model_type** (int) - model类型。支持1为encoder_or_decoder，2为encoder_and_decoder，3为retro_encoder，4为retro_decoder。
        - **layer_type** (int) - layer类型。支持1为encoder，2为decoder，3为retro_encoder，4为retro_decoder，5为retro_decoder_with_retriever, 默认值： ``1`` 。
        - **self_attn_mask_type** (int) - 注意力mask类型。支持1为padding，2为causal，默认值： ``1`` 。
        - **post_norm** (bool) - 是否在转换器块的末尾插入归一化层。默认值： ``True`` 。
        - **pre_process** (bool) - 使用流水线并行时，表明它是否是第一阶段。默认值： ``False`` 。
        - **post_process** (bool) - 使用流水线并行时，表明它是否是最后一个阶段。默认值： ``False`` 。
        - **drop_path_rate** (float) - drop_path rate。当前不支持该参数大于0，默认值： ``0.0`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - attention掩码矩阵。
        - **encoder_output** (Tensor) - 用于交叉注意力的编码器输出张量，当前不支持该参数。默认值： ``None``。
        - **enc_dec_attn_mask** (Tensor) - 编码器-解码器注意力mask张量，当前不支持该参数。默认值： ``None``。
        - **retriever_input** (Tensor) - 检索输入张量，当前不支持该参数。默认值： ``None``。
        - **retriever_output** (Tensor) - 检索输出张量，当前不支持该参数。默认值： ``None``。
        - **retriever_attn_mask** (Tensor) - 检索注意力mask张量，当前不支持该参数。默认值： ``None``。
        - **inference_params** (Tensor) - 推理参数的张量，当前不支持该参数。默认值： ``None``。
        - **rotary_pos_emb** (Tensor) - 旋转位置嵌入张量。默认值： ``None``。

    输出：
        - **hidden_states** (Tensor) - 输出张量形状为 :math:`(B, S, H)`。

    异常：
        - **NotImplementedError** - 如果 `drop_path_rate` 大于 0。
        - **NotImplementedError** - 如果 `config` 中的 `distribute_saved_activations` 是 true 并且 `config` 中的 `sequence_parallel` 是 false。
        - **NotImplementedError** - 如果 `config` 中的 `transformer_impl` 是 'transformer_engine'。
        - **NotImplementedError** - 如果 `config` 中的 `fp8` 不是 None。
        - **NotImplementedError** - 如果 `config` 中的 `retro_add_retriever` 是 true。
        - **NotImplementedError** - 如果 `model_type` 是 3 或 4。
        - **NotImplementedError** - 如果 `encoder_output`、`enc_dec_attn_mask`、`retriever_input`、`retriever_output`、`retriever_attn_mask` 或 `inference_params` 不是 None。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst