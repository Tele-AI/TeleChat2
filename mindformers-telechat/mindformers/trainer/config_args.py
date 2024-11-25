# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindFormers' Config API."""
import os
from typing import Optional, Union
from dataclasses import dataclass
import inspect
from mindformers.core.callback import CheckpointMonitor
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType
from mindformers.tools.utils import get_real_group_size


__all__ = ['BaseArgsConfig', 'RunnerConfig', 'DatasetConfig', 'DataLoaderConfig',
           'ConfigArguments', 'ContextConfig', 'CloudConfig', 'CheckpointConfig',
           'ParallelContextConfig', 'OptimizerConfig', 'LRConfig', 'WrapperConfig']


@dataclass
class BaseArgsConfig:
    """Base Argument config."""
    _support_kwargs = []

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key not in self._support_kwargs:
                    raise ValueError(f"The Config Class support input argument is {self._support_kwargs}"
                                     f", but got {key}.")
                if value is None:
                    continue
                if isinstance(value, BaseArgsConfig):
                    value = value.__dict__
                self.__setattr__(key, value)


@dataclass
class ContextConfig(BaseArgsConfig):
    r"""Context Config For Running Environment.

    Context should be configured before running your program. If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        The mode is not recommended to be changed after net was initialized because the implementations of some
        operations are different in graph mode and pynative mode. Default: ``PYNATIVE_MODE`` .

    Some configurations are device specific, see the below table for details:

    +-------------------------+------------------------------+----------------------------+
    | Function Classification |   Configuration Parameters   |   Hardware Platform Support|
    +=========================+==============================+============================+
    | System Configuration    |   device_id                  |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |   device_target              |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  max_device_memory           |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  variable_memory_max_size    |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  mempool_block_size          |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  op_timeout                  |  Ascend                    |
    +-------------------------+------------------------------+----------------------------+
    | Debug Configuration     |  save_graphs                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  save_graphs_path            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_dump                 |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  save_dump_path              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  deterministic               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  print_file_path             |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  env_config_path             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  precompile_only             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  reserve_class_name_in_scope |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  pynative_synchronize        |  GPU/Ascend                |
    +-------------------------+------------------------------+----------------------------+
    | Executive Control       |   mode                       |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  enable_graph_kernel         |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  enable_reduce_precision     |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  check_bprop                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  max_call_depth              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  grad_for_scalar             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_compile_cache        |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  inter_op_parallel_num       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  runtime_num_threads         |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  compile_cache_path          |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  disable_format_transform    |  GPU                       |
    |                         +------------------------------+----------------------------+
    |                         |  support_binary              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_optimize_level       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_offload              |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  ascend_config               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  jit_syntax_level            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  gpu_config                  |  GPU                       |
    +-------------------------+------------------------------+----------------------------+

    Args:
        device_id (int):
            ID of the target device, the value must be in [0, device_num_per_host-1],
            while device_num_per_host should be no more than 4096. Default: ``0`` .
        device_target (str):
            The target device to run, support "Ascend", "GPU", and "CPU".
            If device target is not set, the version of MindSpore package is used.
        max_device_memory (str):
            Set the maximum memory available for devices. The format is "xxGB".
            Default: ``" 1024GB"`` . The actual used memory size is the minimum of the available memory of the device
            and max_device_memory.
        variable_memory_max_size (str):
            This parameter is deprecated, and will be removed in a future version.
            Please use parameter 'max_device_memory' instead.
        mempool_block_size (str):
            Set the size of the memory pool block in PyNative mode for devices.
            The format is "xxGB". Default: ``"1GB"`` . Minimum size is "1G". The actual used memory block size is the
            minimum of the available memory of the device and mempool_block_size.
        op_timeout (int):
            Set the maximum duration of executing an operator in seconds.
            If the execution time exceeds this value, system will terminate the task. 0 means endless wait.
            Default: ``1900`` .
        save_graphs (bool or int):
            Whether to save intermediate compilation graphs. Default: ``0`` .
            Available values are:

            - False or 0: disable saving of intermediate compilation graphs.
            - 1: some intermediate files will be generated during graph compliation.
            - True or 2: Generate more ir files related to backend process.
            - 3: Generate visualization computing graphs and detailed frontend ir graphs.

            When the `save_graphs` attribute is set as ``True`` , ``1`` , ``2`` or ``3`` , attribute of
            `save_graphs_path` is used to set the intermediate compilation graph storage path. By default, the graphs
            are saved in the current directory.
        save_graphs_path (str):
            Path to save graphs. Default: ".".
            If the specified directory does not exist, the system will automatically create the directory.
            During distributed training, graphs will be saved to the directory of
            `save_graphs_path/rank_${rank_id}/`. `rank_id` is the ID of the current device in the cluster.
        deterministic (str):
            Whether to enable op run in deterministic mode. The value must be in the
            range of ['ON', 'OFF'], and the default value is ``'OFF'`` .

            - "ON": Enable operator deterministic running mode.
            - "OFF": Disable operator deterministic running mode.

            When deterministic mode is on, model ops will be deterministic in Ascend. This means that if op run
            multiple times with the same inputs on the same hardware, it will have the exact same outputs each time.
            This is useful for debugging models.
        enable_dump (bool):
            This parameters is deprecated, and will be deleted in the next version.
        save_dump_path (str):
            This parameters is deprecated, and will be deleted in the next version.
        print_file_path (str):
            The path of saving print data. If this parameter is set, print data is saved to
            a file by default, and print_file_path is not set, the screen will be displayed.
            If the saved file already exists, the timestamp suffix will be added to the file. Saving data to a file
            solves the problem of data loss in screen printing when a large amount of data is generated.
            If it is not set, an error will be reported: prompt to set the upper absolute path.
        env_config_path (str): Config path for DFX.
            Through mindspore.set_context(env_config_path="./mindspore_config.json")

            configure RDR:

            - enable: controls whether the RDR is enabled to collect the key data during training and
              save key data in the fault scenario. When set to ``true`` , the RDR will be turned on.
              When set to ``false`` , the RDR will be turned off.
            - mode: sets the mode of RDR on exporting data. When set to ``1`` , the RDR only exports data
              in the fault scenario. When set to ``2`` , the RDR exports data in the fault scenario and the
              normal end scenario. Default: ``1`` .
            - path: sets the path where RDR saves data. The current path must be absolute.

            Memory reuse:

            - mem_Reuse: controls whether the memory reuse function is turned on. When set to ``True`` ,
              the memory reuse function is turned on. When set to ``False`` , the memory reuse function is turned off.

        precompile_only (bool):
            Whether to only precompile the network. Default: ``False`` .
            If set to ``True`` , the network will only be compiled, not executed.
        reserve_class_name_in_scope (bool) :
            Whether to save the network class name in the scope. Default: ``True`` .
            Each node has a scope. A scope of a subnode is the name of its parent node. If reserve_class_name_in_scope
            is set to ``True`` , the class name will be saved after keyword 'net-' in the scope.
            For example:

            Default/net-Net1/net-Net2 (reserve_class_name_in_scope=True)

            Default/net/net (reserve_class_name_in_scope=False)

        pynative_synchronize (bool):
            Whether to enable synchronous execution of the device in PyNative mode.
            Default: ``False`` . When the value is set to ``False`` , the operator is executed asynchronously on the
            device. When an error occurs in the execution of the operator, the specific error script code location
            cannot be located, when the value is set to ``True`` , the operator is executed synchronously on the
            device. It will reduce the execution performance of the program. At this time, when an error occurs in the
            execution of the operator, the location of the error script code can be located according to the call stack
            of the error.
        mode (int):
            Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).
            Both modes support all backends. Default: ``PYNATIVE_MODE`` .
        enable_graph_kernel (bool):
            Whether to enable graph kernel fusion to optimize network execution performance.
            Default: ``False`` .
            Indicates whether to enable image-computing convergence to optimize network execution performance.
            If enable_graph_kernel is set to ``True`` , acceleration can be enabled.
            For details of graph kernel fusion, please check
            `Enabling Graph Kernel Fusion
            <https://www.mindspore.cn/tutorials/experts/en/r2.1/optimize/graph_fusion_engine.html>`_.
        enable_reduce_precision (bool):
            Whether to enable precision reduction.
            If the operator does not support the user-specified precision, the precision will
            be changed automatically. Default: ``True`` .
        check_bprop (bool):
            Whether to check back propagation nodes. The checking ensures that the shape and dtype
            of back propagation node outputs is the same as input parameters. Default: ``False`` .
        max_call_depth (int):
            Specify the maximum depth of function call. Must be positive integer. Default: ``1000`` .
            The max_call_depth parameter needs to be set when the nested call is too deep or the number
            of subgraphs is too large. If max_call_depth is set larger than before, the system max stack depth should be
            set larger too, otherwise a `core dumped` exception may be raised because of system stack overflow.
        grad_for_scalar (bool):
            Whether to get gradient for scalar. Default: ``False`` .
            When grad_for_scalar is set to ``True`` , the function's scalar input can be derived.
            The default value is ``False`` . Because the back-end does not support scaling operations currently,
            this interface only supports simple operations that can be deduced by the front-end.
        enable_compile_cache (bool):
            Whether to save or load the cache of the graph compiled by front-end.
            After enable_compile_cache is set to ``True`` , during the first execution, a hardware-independent
            compilation cache is generated and exported to a MINDIR file. When the network is executed again,
            if enable_compile_cache is still set to ``True`` and the network scripts are not changed,
            the compile cache is loaded. Note that only limited automatic detection for the changes of
            python scripts is supported by now, which means that there is a correctness risk. Default: ``False`` .
            This is an experimental prototype that is subject to change and/or deletion.
        compile_cache_path (str):
            Path to save the compile cache. Default: ".".
            If the specified directory does not exist, the system will automatically create the directory.
            The cache will be saved to the directory of `compile_cache_path/rank_${rank_id}/`. The `rank_id` is
            the ID of the current device in the cluster.
        inter_op_parallel_num(int):
            The thread number of op parallel at the same time. Default value is ``0`` ,
            which means use the default num.
        runtime_num_threads(int):
            The thread pool number of cpu kernel used in runtime,
            which must bigger than or equal to 0. Default value is ``30`` , if you run many processes at
            the same time, you should set the value smaller to avoid thread contention.
        disable_format_transform (bool):
            Whether to disable the automatic format transform function from NCHW to NHWC.
            When the network training performance of fp16 is worse than fp32, `disable_format_transform` can be set to
            ``True`` to try to improve training performance. Default: ``False`` .
        support_binary (bool):
            Whether to support run .pyc or .so in graph mode. If want to support run .so or .pyc
            in graph mode, coulde set 'support_binary' to be ``True`` , and run once .py file. It would save the source
            of the interfaces would be compiled by MindSpore to the interfaces definition .py file that should be
            guaranteed to be writable. Then compile the .py file to the .pyc or .so file, and could run in Graph mode.
        memory_optimize_level (str):
            The memory optimize level.
            Default: O0. The value must be in ['O0', 'O1'].

            - O0: priority performance option, disable SOMAS (Safe Optimized Memory Allocation Solver).
            - O1: priority memory option, enable SOMAS.
        memory_offload (str):
            Whether to enable the memory offload function. When it is enabled, the idle data will be
            temporarily copied to the host side in the case of insufficient device memory. The value must be in the
            range of ['ON', 'OFF'], and the default value is ``'OFF'`` .

            - ON: Enable the memory Offload function. On Ascend hardware platform, this parameter does not take effect
              when the environment variable "GRAPH_OP_RUN=1" is not set; This parameter does not take effect when
              memory_optimize_level is set 'O1'.
            - OFF: Turn off the memory Offload function.
        ascend_config (dict):
            Set the parameters specific to Ascend hardware platform. It is not set by default.
            Currently, configurations except `parallel_speed_up_json_path` are currently only supported on
            Atlas 800T A2 hardware platform. The default value of `precision_mode`, `jit_compile` and
            `atomic_clean_policy` are experimental parameters, may change in the future.

            - precision_mode (str): Mixed precision mode setting, on Atlas 800T A2 hardware platform, the default
              value of training network is based on the value of CANN, and the default value of inference network
              is ``force_fp16`` . The value range is as follows:

              - force_fp16: When the operator supports both float16 and float32, select float16 directly.
              - allow_fp32_to_fp16: When the operator does not support the float32 data type, directly reduce
                the precision of float16.
              - allow_mix_precision: Automatic mixing precision, facing the whole network operator, according
                to the built-in optimization strategy, automatically reduces the precision of some operators
                to float16 or bfloat16.
              - must_keep_origin_dtype: Keep the accuracy of the original drawing.
              - force_fp32: When the input of the matrix calculation operator is float16 and the output supports
                float16 and float32, output is forced to float32.
              - allow_fp32_to_bf16: When the operator does not support the float32 data type, directly reduce
                the precision of bfloat16.
              - allow_mix_precision_fp16: Automatic mixing precision, facing the whole network operator, automatically
                reduces the precision of some operators to float16 according to the built-in optimization strategy.
              - allow_mix_precision_bf16: Automatic mixing precision, facing the whole network operator, according to
                the built-in optimization strategy, automatically reduces the precision of some operators to bfloat16.

            - jit_compile (bool): Whether to select online compilation. the default value is based on CANN.
            - atomic_clean_policy (int): The policy for cleaning memory occupied by atomic operators in the network.
              Default: ``1`` .

              - 0: The memory occupied by all atomic operators in the network is cleaned centrally.
              - 1: Memory is not cleaned centrally and each atomic operator in the network is cleaned separately.
                When the memory of the network exceeds the limit, you may try this cleaning policy, but it may cause
                performance loss.
            - matmul_allow_hf32 (bool): Whether to convert FP32 to HF32 for Matmul operators. Default value: ``False``.
              For detailed information, please refer to `Ascend community <https://www.hiascend.com/>`_ .
            - conv_allow_hf32 (bool): Whether to convert FP32 to HF32 for Conv operators. Default value: ``True``.
              For detailed information, please refer to `Ascend community <https://www.hiascend.com/>`_ .
            - op_precision_mode (str): Path to config file of op precision mode. For detailed information, please refer
              to `Ascend community <https://www.hiascend.com/>`_ .
            - parallel_speed_up_json_path(Union[str, None]): The path to the parallel speed up json file, configuration
              can refer to `parallel_speed_up.json
              <https://gitee.com/mindspore/mindspore/blob/r2.1/config/parallel_speed_up.json>`_ .
              If its value is None or '', it does not take effect. Default None.

              - recompute_comm_overlap (bool): Enable overlap between recompute ops and communication ops if True.
                Default: False.
              - matmul_grad_comm_overlap (bool): Enable overlap between grad ops and communication ops if True.
                Default: False.
              - enable_task_opt (bool): Enable the optimizaton of the number of tasks for each communication if True.
                Default: False.
              - interleaved_matmul_comm (bool): Enable interleaved optimization of Matmul-Comm if True. Default: False.
              - interleaved_layernorm_comm (bool): Enable interleaved optimization of LayerNorm-Comm if True.
                Default: False.
        jit_syntax_level (int):
            Set JIT syntax level for graph compiling, triggered by GRAPH_MODE and @jit decorator.
            The value must be in [STRICT, LAX]. Default: LAX. All levels
            support all backends.

            - STRICT: Only basic syntax is supported, and execution performance is optimal.
            - LAX: Compatible with all Python syntax as much as possible. However, execution performance may be
              affected and not optimal.
        gpu_config (dict):
            Set the parameters specific to gpu hardware platform. It is not set by default.
            Currently, only setting `conv_fprop_algo` and `conv_dgrad_algo` and `conv_wgrad_algo` are supported on GPU
            hardware platform.

            - conv_fprop_algo (str): Specifies convolution forward algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - implicit_gemm: This algorithm expresses the convolution as a matrix product without actually explicitly
                forming the matrix that holds the input tensor data.
              - implicit_precomp_gemm: This algorithm expresses convolution as a matrix product without actually
                explicitly forming the matrix that holds the input tensor data, but still needs some memory workspace to
                precompute some indices in order to facilitate the implicit construction of the matrix that holds the
                input tensor data.
              - gemm: This algorithm expresses the convolution as an explicit matrix product. A significant memory
                workspace is needed to store the matrix that holds the input tensor data.
              - direct: This algorithm expresses the convolution as a direct convolution (for example, without
                implicitly or explicitly doing a matrix multiplication).
              - fft: This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft algorithm for
                large size images.
              - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
                sized workspace is needed to store intermediate results.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution. A
                significant workspace may be needed to store intermediate results.
            - conv_dgrad_algo (str): Specifies convolution data grad algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
                forming the matrix that holds the input tensor data. The sum is done using the atomic add operation,
                thus the results are non-deterministic.
              - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
                the matrix that holds the input tensor data. The results are deterministic.
              - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results. The results are deterministic.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft for large size
                images. The results are deterministic.
              - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
                sized workspace is needed to store intermediate results. The results are deterministic.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution.
                A significant workspace may be needed to store intermediate results. The results are deterministic.
            - conv_wgrad_algo (str): Specifies convolution filter grad algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
                forming the matrix that holds the input tensor data. The sum is done using the atomic add operation,
                thus the results are non-deterministic.
              - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
                the matrix that holds the input tensor data. The results are deterministic.
              - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results. The results are deterministic.
              - algo_3: This algorithm is similar to algo_0 but uses some small workspace to precompute some indices.
                The results are also non-deterministic.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution.
                A significant workspace may be needed to store intermediate results. The results are deterministic.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft for large size
                images. The results are deterministic.

    Raises:
        ValueError: If input key is not an attribute in context.
    """
    _support_kwargs = [
        'mode', 'precompile_only', 'device_target', 'device_id', 'save_graphs',
        'save_graphs_path', 'enable_dump', 'auto_tune_mode',
        'save_dump_path', 'enable_reduce_precision', 'variable_memory_max_size',
        'enable_profiling', 'profiling_options', 'enable_auto_mixed_precision',
        'enable_graph_kernel', 'reserve_class_name_in_scope', 'check_bprop',
        'max_device_memory', 'print_file_path', 'enable_sparse', 'max_call_depth',
        'env_config_path', 'save_compile_cache', 'runtime_num_threads',
        'load_compile_cache', 'grad_for_scalar', 'pynative_synchronize', 'mempool_block_size'
    ]

    def __init__(self,
                 mode: Optional[Union[int, str]] = 0,
                 device_target: str = "Ascend",
                 device_id: int = int(os.getenv('DEVICE_ID', '0')),
                 save_graphs: bool = False, save_graphs_path: str = ".", **kwargs):
        super(ContextConfig, self).__init__(mode=mode,
                                            device_id=device_id,
                                            device_target=device_target,
                                            save_graphs=save_graphs,
                                            save_graphs_path=save_graphs_path, **kwargs)


@dataclass
class ParallelContextConfig(BaseArgsConfig):
    r"""Parallel Context Config.
    Set auto parallel context, only data parallel supported on CPU.

    Note:
        Attribute name is required for setting attributes.
        If a program has tasks on different parallel modes, before setting a new parallel mode for the
        next task, interface :func:`mindspore.reset_auto_parallel_context` should be called to reset
        the configuration.
        Setting or changing parallel modes must be called before creating any Initializer, otherwise,
        it may have RuntimeError when compiling the network.

    Some configurations are parallel mode specific, see the below table for details:

    ===========================  ===========================
    Common                       AUTO_PARALLEL
    ===========================  ===========================
    device_num                   gradient_fp32_sync
    global_rank                  loss_repeated_mean
    gradients_mean               search_mode
    parallel_mode                strategy_ckpt_load_file
    all_reduce_fusion_config     strategy_ckpt_save_file
    enable_parallel_optimizer    dataset_strategy
    parallel_optimizer_config    pipeline_stages
    enable_alltoall              auto_parallel_search_mode
               \                 comm_fusion
               \                 strategy_ckpt_config
    ===========================  ===========================

    Args:
        device_num (int):
            Available device number, the value must be in [1, 4096]. Default: ``1`` .
        global_rank (int):
            Global rank id, the value must be in [0, 4095]. Default: ``0`` .
        gradients_mean (bool):
            Whether to perform mean operator after allreduce of gradients.
            "stand_alone" do not support gradients_mean. Default: ``False`` .
        gradient_fp32_sync (bool):
            Run allreduce of gradients in fp32. "stand_alone", "data_parallel"
            and "hybrid_parallel" do not support gradient_fp32_sync. Default: ``True`` .
        parallel_mode (str):
            There are five kinds of parallel modes, ``"stand_alone"`` , ``"data_parallel"`` ,
            ``"hybrid_parallel"`` , ``"semi_auto_parallel"`` and ``"auto_parallel"`` . Note the pynative mode
            only supports the ``"stand_alone"`` and ``"data_parallel"`` mode. Default: ``"stand_alone"`` .

                 - stand_alone: Only one processor is working.

                 - data_parallel: Distributes the data across different processors.

                 - hybrid_parallel: Achieves data parallelism and model parallelism manually.

                 - semi_auto_parallel: Achieves data and model parallelism by setting parallel strategies.

                 - auto_parallel: Achieving parallelism automatically.
        search_mode (str):
            There are three kinds of shard strategy search modes: ``"recursive_programming"`` ,
            ``"dynamic_programming"`` and ``"sharding_propagation"`` . Default: ``"dynamic_programming"`` .

                 - recursive_programming: Recursive programming search mode. In order to obtain optimal performance,
                   it is recommended that users set the batch size to be greater than or equal to the product of
                   the number of devices and the number of multi-copy parallelism.

                 - dynamic_programming: Dynamic programming search mode.

                 - sharding_propagation: Propagate shardings from configured ops to non-configured ops.
        auto_parallel_search_mode (str):
            This is the old version of 'search_mode'. Here, remaining this attribute is
            for forward compatibility, and this attribute will be deleted in a future MindSpore version.
        parameter_broadcast (bool):
            Whether to broadcast parameters before training. Before training, in order to have
            the same network initialization parameter values for all devices, broadcast the parameters
            on device 0 to other devices. Parameter broadcasting in different parallel modes is different,
            ``data_parallel`` mode, all parameters are broadcast except for the parameter whose attribute
            layerwise_parallel is ``True`` . ``Hybrid_parallel`` , ``semi_auto_parallel``  and
            ``auto_parallel mode`` , the segmented parameters do not participate in broadcasting.
            Default: ``False`` .
        strategy_ckpt_load_file (str): The path to load parallel strategy checkpoint. The parameter is not to be
                       recommended currently, it is better using 'strategy_ckpt_config' to replace it. Default: ``''``
        strategy_ckpt_save_file (str): The path to save parallel strategy checkpoint. The parameter is not to be
                       recommended currently, it is better using 'strategy_ckpt_config' to replace it. Default: ``''``
        full_batch (bool): If you load whole batch datasets in ``auto_parallel`` mode, this parameter
                       should be set as ``True`` . Default: ``False`` . The interface is not to be recommended
                       currently, it is better using 'dataset_strategy' to replace it.
        dataset_strategy (Union[str, tuple]): Dataset sharding strategy. Default: ``"data_parallel"`` .
                       dataset_strategy="data_parallel" is equal to full_batch=False, dataset_strategy="full_batch" is
                       equal to full_batch=True. For execution mode is 'GRAPH_MODE' and dataset load into net by model
                       parallel strategy likes ds_stra ((1, 8), (1, 8)), it requires using
                       set_auto_parallel_context(dataset_strategy=ds_stra).
        enable_parallel_optimizer (bool): This is a developing feature, which shards the weight update computation for
                       data parallel training in the benefit of time and memory saving. Currently, auto and semi auto
                       parallel mode support all optimizers in both Ascend and GPU. Data parallel mode only supports
                       `Lamb` and `AdamWeightDecay` in Ascend . Default: ``False`` .
        enable_alltoall (bool): A switch that allows AllToAll operators to be generated during communication. If its
                        value is ``False`` , there will be a combination of operators such as AllGather, Split and
                        Concat instead of AllToAll. Default: ``False`` .
        all_reduce_fusion_config (list): Set allreduce fusion strategy by parameters indices. Only support ReduceOp.SUM
                       and HCCL_WORLD_GROUP/NCCL_WORLD_GROUP. No Default, if it is not set, the fusion is closed.
        pipeline_stages (int): Set the stage information for pipeline parallel. This indicates how the devices are
                        distributed alone in the pipeline. The total devices will be divided into 'pipeline_stags'
                        stages. Default: ``1`` .
        grad_accumulation_step (int): This interface is deprecated. Default: ``1`` .
        parallel_optimizer_config (dict):
            A dict contains the keys and values for setting the parallel optimizerconfigure.
            The configure provides more detailed behavior control about parallel training
            when parallel optimizer is enabled. Currently it supports the key `gradient_accumulation_shard`.
            The configure will be effective when we use
            mindspore.set_auto_parallel_context(enable_parallel_optimizer=True).
            It supports the following keys.

                - gradient_accumulation_shard(bool): If ``true`` , the accumulation gradient parameters will be
                  sharded across the data parallel devices.
                  This will introduce additional communication(ReduceScatter)
                  at each step when accumulate the gradients,
                  but saves a lot of device memories, thus can make model be trained with larger batch size.
                  This configure is effective only when the model runs on pipeline training or gradient
                  accumulation with data parallel. Default ``False`` .

                - parallel_optimizer_threshold(int): Set the threshold of parallel optimizer. When parallel
                  optimizer is enabled, parameters with size smaller than this threshold will not be sharded
                  across the devices. Parameter size = shape[0] \* ... \* shape[n] \* size(dtype). Non-negative.
                  Unit: KB. Default: ``64`` .

        comm_fusion (dict):
            A dict contains the types and configurations for setting the communication fusion.
            Each communication fusion config has two keys: "mode" and "config".
            It supports following communication fusion types and configurations:

                - openstate: Whether turn on the communication fusion or not. If `openstate` is ``True`` ,
                  turn on the communication fusion, otherwise, turn off the communication fusion.
                  Default: ``True`` .

                - allreduce: If communication fusion type is `allreduce`. The `mode` contains: `auto`, `size`
                  and `index`. In `auto` mode, AllReduce fusion is configured by gradients size and the default
                  fusion threshold is `64` MB. In 'size' mode, AllReduce fusion is configured by gradients size
                  manually, and the fusion threshold must be larger than `0` MB. In `index` mode, it is same as
                  `all_reduce_fusion_config`.

                - allgather: If communication fusion type is `allgather`. The `mode` contains: `auto`, `size`.
                  In `auto` mode, AllGather fusion is configured by gradients size, and the default fusion
                  threshold is `64` MB. In 'size' mode, AllGather fusion is configured by gradients size
                  manually, and the fusion threshold must be larger than `0` MB.

                - reducescatter: If communication fusion type is `reducescatter`. The `mode` contains: `auto`
                  and `size`. Config is same as `allgather`.

        strategy_ckpt_config (dict):
            A dict contains the configurations for setting the parallel strategy file.
            This interface contains the functions of parameter `strategy_ckpt_load_file` and
            `strategy_ckpt_save_file`, it is recommonded to use this parameter to replace those two
            parameters. It contains following configurations:

                - load_file (str): The path to load parallel strategy checkpoint. If the file name extension is
                  `.json`, the file is loaded in JSON format. Otherwise, the file is loaded in ProtoBuf
                  format.
                  Default: ''

                - save_file (str): The path to save parallel strategy checkpoint. If the file name extension is
                  `.json`, the file is saved in JSON format. Otherwise, the file is saved in ProtoBuf format.
                  Default: ''

                - only_trainable_params (bool): Only save/load the strategy information for trainable parameter.
                  Default: ``True`` .

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """

    _support_kwargs = [
        'device_num', 'global_rank', 'gradients_mean', 'gradient_fp32_sync', 'parallel_mode',
        'auto_parallel_search_mode', 'search_mode', 'parameter_broadcast', 'strategy_ckpt_load_file',
        'strategy_ckpt_save_file', 'full_batch', 'enable_parallel_optimizer', 'enable_alltoall',
        'all_reduce_fusion_config', 'pipeline_stages', 'grad_accumulation_step',
        'parallel_optimizer_config', 'comm_fusion'
    ]

    def __init__(self,
                 parallel_mode: str = 'STAND_ALONE',
                 device_num: int = get_real_group_size(),
                 gradients_mean: bool = False, **kwargs):
        super(ParallelContextConfig, self).__init__(parallel_mode=parallel_mode,
                                                    device_num=device_num,
                                                    gradients_mean=gradients_mean, **kwargs)


@dataclass
class CloudConfig(BaseArgsConfig):
    """Cloud Config For ModelArts."""
    _support_kwargs = [
        'obs_path', 'root_path', 'rank_id', 'upload_frequence',
        'keep_last', 'retry', 'retry_time'
    ]

    def __init__(self,
                 obs_path: str = None,
                 root_path: str = '/cache',
                 rank_id: int = None,
                 upload_frequence: int = 1,
                 keep_last: bool = False, **kwargs):
        super(CloudConfig, self).__init__(obs_path=obs_path,
                                          root_path=root_path,
                                          rank_id=rank_id,
                                          upload_frequence=upload_frequence,
                                          keep_last=keep_last, **kwargs)


@dataclass
class RunnerConfig(BaseArgsConfig):
    """MindFormers' config when running model."""

    _support_kwargs = [
        'epochs', 'batch_size', 'sink_mode', 'sink_size', 'initial_epoch',
        'has_trained_epoches', 'has_trained_steps', 'image_size', 'num_classes',
        'sink_size',
    ]

    def __init__(self,
                 epochs: int = None, batch_size: int = None,
                 sink_mode: bool = None, sink_size: int = None,
                 initial_epoch: int = None, has_trained_epoches: int = None,
                 has_trained_steps: int = None, **kwargs):
        super(RunnerConfig, self).__init__(epochs=epochs,
                                           batch_size=batch_size,
                                           sink_mode=sink_mode,
                                           sink_size=sink_size,
                                           initial_epoch=initial_epoch,
                                           has_trained_steps=has_trained_steps,
                                           has_trained_epoches=has_trained_epoches, **kwargs)


@dataclass
class CheckpointConfig(BaseArgsConfig):
    """MindFormers' save checkpoint config."""

    _support_kwargs = inspect.getfullargspec(CheckpointMonitor).args

    def __init__(self,
                 prefix: str = 'mindformers',
                 directory: str = None,
                 save_checkpoint_steps: int = 1,
                 keep_checkpoint_max: int = 1,
                 integrated_save: bool = True,
                 async_save: bool = False,
                 saved_network: bool = None, **kwargs):
        super(CheckpointConfig, self).__init__(prefix=prefix,
                                               directory=directory,
                                               saved_network=saved_network,
                                               save_checkpoint_steps=save_checkpoint_steps,
                                               keep_checkpoint_max=keep_checkpoint_max,
                                               integrated_save=integrated_save,
                                               async_save=async_save, **kwargs)


@dataclass
class LRConfig(BaseArgsConfig):
    """MindFormers' learning rate schedule config."""
    _support_kwargs = [
        'type', 'max_lr', 'min_lr', 'decay_steps', 'decay_rate',
        'power', 'end_learning_rate', 'warmup_steps'
    ]

    def __init__(self, lr_type: str = None, **kwargs):
        if lr_type is not None:
            lr_schedule = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.LR, class_name=lr_type)
            self._support_kwargs.extend(inspect.getfullargspec(lr_schedule).args)
        super(LRConfig, self).__init__(type=lr_type, **kwargs)


@dataclass
class OptimizerConfig(BaseArgsConfig):
    """MindFormers' optimizer config."""
    _support_kwargs = [
        'type', 'learning_rate', 'beta1', 'beta2', 'eps', 'epsilon',
        'weight_decay', 'loss_scale', 'momentum'
    ]

    def __init__(self, optim_type: str = None,
                 learning_rate: Optional[Union[BaseArgsConfig, float]] = None,
                 **kwargs):
        if optim_type is not None:
            optimizer = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.OPTIMIZER, class_name=optim_type)
            self._support_kwargs.extend(inspect.getfullargspec(optimizer).args)

        super(OptimizerConfig, self).__init__(type=optim_type,
                                              learning_rate=learning_rate,
                                              **kwargs)


@dataclass
class WrapperConfig(BaseArgsConfig):
    """MindFormers' wrapper config."""
    _support_kwargs = [
        'type', 'sens', 'scale_sense'
    ]

    def __init__(self, wrapper_type: str = None, **kwargs):
        if wrapper_type is not None:
            wrapper = MindFormerRegister.get_cls(
                module_type=MindFormerModuleType.WRAPPER, class_name=wrapper_type)
            self._support_kwargs.extend(inspect.getfullargspec(wrapper).args)

        super(WrapperConfig, self).__init__(type=wrapper_type, **kwargs)


@dataclass
class DataLoaderConfig(BaseArgsConfig):
    """MindFormers' data loader config."""
    _support_kwargs = [
        'type', 'dataset_dir', 'num_samples', 'num_parallel_workers',
        'shuffle', 'sampler', 'extensions', 'class_indexing', 'language_pair',
        'decode', 'num_shards', 'shard_id', 'cache', 'decrypt', 'task', 'usage',
        'test_set', 'valid_set', 'padded_sample', 'num_padded'
    ]

    def __init__(self, dataloader_type: str = None, dataset_dir: str = None, **kwargs):
        if dataloader_type is not None:
            dataloader = MindFormerRegister.get_cls(
                MindFormerModuleType.DATASET_LOADER, class_name=dataloader_type)
            self._support_kwargs.extend(inspect.getfullargspec(dataloader).args)
        super(DataLoaderConfig, self).__init__(type=dataloader_type,
                                               dataset_dir=dataset_dir,
                                               **kwargs)


@dataclass
class DatasetConfig(BaseArgsConfig):
    """MindFormers' dataset config."""
    _support_kwargs = [
        'data_loader', 'input_columns', 'output_columns', 'column_order',
        'drop_remainder', 'repeat', 'batch_size', 'image_size', 'num_parallel_workers',
        'per_batch_map', 'python_multiprocessing', 'max_rowsize', 'cache', 'offload'
    ]

    def __init__(self,
                 data_loader: Optional[Union[dict, BaseArgsConfig]] = None,
                 input_columns: Optional[Union[str, list]] = None,
                 output_columns: Optional[Union[str, list]] = None,
                 column_order: Optional[Union[str, list]] = None,
                 drop_remainder: bool = True, repeat: int = 1, batch_size: int = None,
                 image_size: Optional[Union[int, list, tuple]] = None, **kwargs):
        super(DatasetConfig, self).__init__(data_loader=data_loader,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            repeat=repeat,
                                            input_columns=input_columns,
                                            output_columns=output_columns,
                                            column_order=column_order,
                                            drop_remainder=drop_remainder, **kwargs)


@dataclass
class ConfigArguments(BaseArgsConfig):
    """MindFormers' config arguments."""
    _support_kwargs = [
        'output_dir', 'profile', 'auto_tune', 'filepath_prefix', 'autotune_per_step',
        'train_dataset', 'eval_dataset', 'predict_dataset', 'runner_config', 'optimizer',
        'lr_schedule', 'save_checkpoint', 'cloud_config', 'seed', 'runner_wrapper'
    ]

    def __init__(self, output_dir: str = './output', profile: bool = False,
                 auto_tune: bool = False, filepath_prefix: str = './autotune',
                 autotune_per_step: int = 10, seed: int = None,
                 train_dataset: Optional[Union[dict, BaseArgsConfig]] = None,
                 eval_dataset: Optional[Union[dict, BaseArgsConfig]] = None,
                 runner_config: Optional[Union[dict, BaseArgsConfig]] = None,
                 optimizer: Optional[Union[dict, BaseArgsConfig]] = None,
                 runner_wrapper: Optional[Union[dict, BaseArgsConfig]] = None,
                 lr_schedule: Optional[Union[dict, BaseArgsConfig]] = None,
                 save_checkpoint: Optional[Union[dict, BaseArgsConfig]] = None,
                 cloud_config: Optional[Union[dict, BaseArgsConfig]] = None):
        super(ConfigArguments, self).__init__(output_dir=output_dir,
                                              profile=profile,
                                              auto_tune=auto_tune,
                                              seed=seed,
                                              filepath_prefix=filepath_prefix,
                                              autotune_per_step=autotune_per_step,
                                              train_dataset=train_dataset,
                                              eval_dataset=eval_dataset,
                                              runner_config=runner_config,
                                              optimizer=optimizer,
                                              runner_wrapper=runner_wrapper,
                                              lr_schedule=lr_schedule,
                                              save_checkpoint=save_checkpoint,
                                              cloud_config=cloud_config)
