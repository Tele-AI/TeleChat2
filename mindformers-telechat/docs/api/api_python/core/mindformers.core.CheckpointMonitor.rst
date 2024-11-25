mindformers.core.CheckpointMonitor
==================================

.. py:class:: mindformers.core.CheckpointMonitor(prefix='CKP', directory=None, config=None, save_checkpoint_steps=1, save_checkpoint_seconds=0, keep_checkpoint_max=5, keep_checkpoint_per_n_minutes=0, integrated_save=True, save_network_params=True, save_trainable_params=False, async_save=False, saved_network=None, append_info=None, enc_key=None, enc_mode='AES-GCM', exception_save=False, global_batch_size=None)

    保存checkpoint的回调函数，训练过程中保存网络参数。

    参数：
        - **prefix** (str) - checkpoint文件的前缀名。默认值： ``CKP`` 。
        - **directory** (str) - checkpoint文件将要保存的文件夹路径。默认值： ``None`` 。
        - **config** (CheckpointConfig) - checkpoint的配置。默认值： ``None`` 。
        - **save_checkpoint_steps** (int) - 每隔多少个step保存一次checkpoint。默认值： ``1`` 。
        - **save_checkpoint_seconds** (int) - 每隔多少秒保存一次checkpoint。不能同时与 `save_checkpoint_steps` 一起使用。默认值： ``0`` 。
        - **keep_checkpoint_max** (int) - 最多保存多少个checkpoint文件。默认值： ``5`` 。
        - **keep_checkpoint_per_n_minutes** (int) - 每隔多少分钟保存一个checkpoint文件。不能同时与 `keep_checkpoint_max` 一起使用。默认值： ``0`` 。
        - **integrated_save** (bool) - 在自动并行场景下，是否合并保存拆分后的Tensor。合并保存功能仅支持在自动并行场景中使用，在手动并行场景中不支持。默认值： ``True`` 。
        - **save_network_params** (bool) - 是否仅额外保存网络参数。默认值： ``True`` 。
        - **save_trainable_params** (bool) - 是否仅额外保存fine-tuned的参数。默认值： ``False`` 。
        - **async_save** (bool) - 是否异步执行保存checkpoint文件。默认值： ``False`` 。
        - **saved_network** (Cell) - 保存在checkpoint文件中的网络。如果 `saved_network` 没有被训练，则保存 `saved_network` 的初始值。默认值： ``None`` 。
        - **append_info** (list) - 保存在checkpoint文件中的信息。支持"epoch_num"、"step_num"和dict类型。dict的key必须是str，dict的value必须是int、float、bool、string、Parameter或Tensor中的一个。默认值： ``None`` 。
        - **enc_key** (Union[None, bytes]) - 用于加密的字节类型key。如果值为None，则不需要加密。默认值： ``None`` 。
        - **enc_mode** (str) - 仅当 `enc_key` 不设为None时，该参数有效。指定了加密模式，目前支持AES-GCM，AES-CBC和SM4-CBC。默认值： ``'AES-GCM'`` 。
        - **exception_save** (bool) - 当有异常发生时，是否保存当前checkpoint文件。默认值： ``False`` 。
        - **global_batch_size** (int) - 总BatchSize大小。默认值： ``0`` 。

    异常：
        - **ValueError** - 如果 `preifx` 不是 `str` 或者包含 `/` 字符。
        - **ValueError** - 如果 `directory` 不是 `str` 。
        - **TypeError** - 如果 `config` 不是 `CheckpointConfig` 类型。