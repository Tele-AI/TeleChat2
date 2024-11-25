mindformers.core.SummaryMonitor
===============================

.. py:class:: mindformers.core.SummaryMonitor(summary_dir=None, collect_freq=10, collect_specified_data=None, keep_default_action=True, custom_lineage_data=None, collect_tensor_freq=None, max_file_size=None, export_options=None)

    SummaryMonitor可以帮助收集收集一些常用信息，比如loss、学习率、计算图等。

    .. note::
        可参考 `note <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.SummaryCollector.html>`_ 。

    参数：
        - **summary_dir** (str) - 收集的数据将存储到此目录。如果目录不存在，将自动创建。默认值： ``None`` 。
        - **collect_freq** (int) - 设置数据收集的频率，频率应大于零，单位为 `step` 。默认值： ``10`` 。
        - **collect_specified_data** (Union[None, dict]) - 对收集的数据进行自定义操作。默认值： ``None`` 。
        - **keep_default_action** (bool) - 此字段影响 `collect_specified_data` 字段的收集行为。默认值： ``True`` 。
        - **custom_lineage_data** (Union[dict, None]) - 允许您自定义数据并将数据显示在MindInsight的 `lineage页面 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/lineage_and_scalars_comparison.html>`_ 。默认值： ``None`` 。
        - **collect_tensor_freq** (Optional[int]) - 语义与 `collect_freq` 的相同，但仅控制TensorSummary。默认值： ``None`` 。
        - **max_file_size** (Optional[int]) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，如果不大于4GB，则设置 `max_file_size=4*1024**3` 。默认值： ``None`` ，表示无限制。
        - **export_options** (Union[None, dict]) - 表示对导出的数据执行自定义操作。默认值： ``None`` ，表示不导出数据。