mindformers.models.multi_modal.ModalContentTransformTemplate
=========================================================================

.. py:class:: mindformers.models.multi_modal.ModalContentTransformTemplate(output_columns: List[str] = None, tokenizer = None, mode = "predict", vstack_columns: List[str] = None, modal_content_padding_size = 1, max_length = 2048, **kwargs)

    转换模态内容模板的基类，可以被特定的模型实现。子类可以通过重写 ``build_conversion_input_text`` 、 ``update_result_before_output`` 、 ``batch`` 、 ``post_process`` 方法来达到模型的期望值。

    参数：
        - **output_columns** (List[str], 可选) - 指定要输出的列。默认值： ``None`` 。
        - **tokenizer** (Tokenizer, 可选) - 构建好的模型tokenizer。默认值： ``None`` 。
        - **mode** (str) - 运行模式，推理 ``predict`` 或者训练 ``train`` 。默认值： ``predict`` 。
        - **vstack_columns** (List[str], 可选) - 指定批处理数据时将使用vstack的列。默认值： ``None`` 。
        - **modal_content_padding_size** (int) - 在训练模式下使用，给继承的 ``Template`` 子类使用，通常表示一个训练样本内支持的模态内容（例如图片）的最大数量，当一个训练样本的模态内容数量小于该值时，会将模态内容扩增至该值。
        - **max_length** (int) - 在训练模式下使用，给继承的Template子类使用，通常表示一个训练样本在分词之后的内容掩码完之后补齐到的最大长度。
        - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

    .. py:method:: batch(data_list, token_padding_length, **kwargs)

        批量处理数据项中的每一列数据。

        参数：
            - **data_list** (list) - 一个包含多个数据项的列表。
            - **token_padding_length** (int) - 用于填充 ``token`` 长度，确保所有文本数据具有相同的长度。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

        返回：
            字典类型，用于存储批量化后的数据。

    .. py:method:: build_conversation_input_text(raw_inputs, result_recorder: DataRecord)
        :classmethod:

        在推理模式下，将传入的数据组装成一个对话，通常被子类继承使用。

        参数：
            - **raw_inputs** (str) - 输入的数据。
            - **result_recorder** (DataRecord) -  结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。

        返回：
            字符串类型，一个组装好的对话。

    .. py:method:: build_labels(text_id_list, result_recorder, **kwargs)

        在训练模式下使用，给继承的子类使用，用于从文本数据中构建训练时所需的标签。

        参数：
            - **text_id_list** (list) - 包含文本数据标识符或索引的列表。
            - **result_recorder** (DataRecord) -  结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

    .. py:method:: build_modal_context(input_ids, result_recorder: DataRecord, **kwargs)

        根据模态生成器的要求，对输入的数据进行处理，最终返回经过处理的数据。

        参数：
            - **input_ids** (list) - 输入的数据。
            - **result_recorder** (DataRecord) -  结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

        返回：
            列表类型，经过处理后的数据。

    .. py:method:: get_need_update_output_items(result: DataRecord)

        获取需要更新的输出项。

        参数：
            - **result** (DataRecord) - 结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。

        返回：
            字典类型，默认为一个空字典。

    .. py:method:: post_process(output_ids, **kwargs)

        将模型输出的序列解码为文本字符串。

        参数：
            - **output_ids** (list) -  一个包含模型输出序列的列表。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

        返回：
            列表类型，包含所有解码后的文本字符串。

    .. py:method:: process_predict_query(query_ele_list: List[Dict], result_recorder: DataRecord)

        在推理模式下，通过遍历找到相应的模态构建器并对其进行处理。

        参数：
            - **query_ele_list** (List[dict]) - 一个预测请求的元素列表，形式如： ``[{"image":"/path/to/image"}, {"text":"describe image in English"}]`` 。
            - **result_recorder** (DataRecord) - 结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。

        返回：
            数组类型，经过每个模态生成器处理过的文本结果。

    .. py:method:: process_train_item(conversation_list: List[List], result_recorder: DataRecord)

        在训练模式下，通过遍历找到相应的模态构建器并对其进行处理。

        参数：
            - **conversation_list** (List[List]) - 一个对话数据的元素列表，形式如： ``[["user", "<img>/path/to/image<img>describe image in English:"], ["assistant", "the image describe ...."]]`` 。
            - **result_recorder** (DataRecord) -  结果数据记录器，用于记录在推理过程中想要保存的数据，数值通过调用 ``DataRecord`` 的 ``put`` 方法进行数据存储。

        返回：
            数组类型，经过每个模态生成器处理过的文本结果。

    .. py:method:: supported_modal()
        :classmethod:

        用于返回一个实例所支持的模态生成器的类型。

        返回：
            列表类型，包含一个实例所支持的模态生成器的类型。