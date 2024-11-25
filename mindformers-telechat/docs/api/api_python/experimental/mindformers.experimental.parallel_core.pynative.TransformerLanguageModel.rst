mindformers.experimental.parallel_core.pynative.TransformerLanguageModel
========================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.TransformerLanguageModel(config, encoder_attn_mask_type, num_tokentypes=0, add_encoder=True, add_decoder=False, decoder_attn_mask_type=AttnMaskType.causal, add_pooler=False, pre_process=True, post_process=True, visual_encoder=None)

    Transformer语言模型。

    参数：
        - **config** (TransformerConfig) – Transformer模型配置，包括初始化函数和并行参数配置等。
        - **encoder_attn_mask_type** (int) – 编码器注意力掩码类型。
        - **num_tokentypes** (int) – 如果大于0，则使用tokentype嵌入。
        - **add_encoder** (bool) – 如果为 ``True`` ，使用编码器。
        - **use_decoder** (bool) – 如果为 ``True`` ，使用解码器。
        - **decoder_attn_mask_type** (int) – 解码器注意力掩码类型。
        - **add_pooler** (bool) – 如果为 ``True`` ，使用池化层。
        - **pre_process** (bool) – 使用流水线并行时，标记它是否为第一阶段。
        - **post_process** (bool) – 使用流水线并行时，标记它是否为最后的阶段。
        - **visual_encoder** (nn.Cell) – 视觉编码器。

    输入：
        - **enc_input_ids** (Tensor) - 编码器输入索引。形状为 :math:`(B, S)` 。
        - **enc_position_ids** (Tensor) - 编码器位置偏移量。形状为 :math:`(B, S)` 。
        - **enc_attn_mask** (Tensor) - 编码器注意力掩码。形状为 :math:`(B, S)` 。
        - **dec_input_ids** (Tensor) - 解码器输入索引。形状为 :math:`(B, S)` 。
        - **dec_position_ids** (Tensor, 可选) - 解码器输入位置索引。形状为 :math:`(B, S)` 。
        - **dec_attn_mask** (Tensor, 可选) - 解码器注意力掩码。形状为 :math:`(B, S)` 。
        - **retriever_input_ids** (Tensor, 可选) - 检索器输入标记索引。
        - **retriever_position_ids** (Tensor, 可选) - 检索器输入位置索引。
        - **retriever_attn_mask** (Tensor, 可选) - 检索器注意力掩码，用于控制在检索器中计算注意力时的注意范围。
        - **enc_dec_attn_mask** (Tensor, 可选) - 编码器-解码器注意力掩码，用于在编码器和解码器之间计算注意力时使用。
        - **tokentype_ids** (Tensor, 可选) - 给模型输入的标记类型索引列表。形状为 :math:`(B, S)` 。
        - **inference_params** (InferenceParams, 可选) - 推理参数，用于在推理过程中指定特定设置，如最大生成长度、最大批处理大小等。
        - **pooling_sequence_index** (int, 可选) - 池化序列索引。
        - **enc_hidden_states** (Tensor, 可选) - 编码器隐藏层。
        - **output_enc_hidden** (bool, 可选) - 是否输出编码器隐藏层。
        - **input_image** (Tensor, 可选) - 输入图像的张量。形状为 :math:`(N, C_{in}, H_{in}, W_{in})` 或 :math:`(N, H_{in}, W_{in}, C_{in}, )` 。
        - **delimiter_position** (Tensor, 可选) - 分隔符位置张量。形状为 :math:`(B, N)` ，其中 :math:`N` 表示分隔符数量。
        - **image_embedding** (Tensor, 可选) - 图像嵌入张量，维度依赖于图像嵌入的维数。

    输出：
        - **encoder_output** (Tensor) - 形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的Tensor。

    异常：
        - **ValueError** – 如果 `config.untie_embeddings_and_output_weights` 且 `add_decoder` 为 ``True`` 。
        - **RuntimeError** – 如果 `input_tensor` 长度为 `1` 。
        - **NotImplementedError** – 如果 `config.retro_add_retriever` 为 ``True`` 。
        - **NotImplementedError** – 如果 `visual_encoder` 或者 `add_decoder` 为 ``True`` 。
        - **NotImplementedError** – 如果 `dec_input_ids` 、 `dec_position_ids` 、 `dec_attn_mask` 、 `retriever_input_ids` 、 `retriever_position_ids` 、 `retriever_attn_mask` 、 `enc_dec_attn_mask` 、 `input_image` 、 `delimiter_position` 或者 `image_embedding` 不为 ``None`` 。
        - **NotImplementedError** – 如果 `output_enc_hidden` 为 ``True`` 。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst
