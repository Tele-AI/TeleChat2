# Config和API组件

## config概念

config包含模型配置、训练配置、环境配置等信息。启动训练、微调、评估和推理时，会先根据config中的内容进行配置。

## config设计

![](https://foruda.gitee.com/images/1691027467343740998/29a1db43_11500692.png)

MindformerConfig是Mindformers整体的配置类，里面包含Mindformers中的各种配置。

- run_xxx.yaml中的配置为默认配置。默认情况下，Mindformers根据yaml配置文件创建MindFormerConfig。
- 如果使用run_mindformer.py方式启动，可以通过命令行参数修改配置中的参数。
- 如果使用Trainer高阶接口方式启动，可以通过TrainArguments修改配置中的参数。

MindformerConfig中包含的具体配置可以参考[Config.md](https://gitee.com/mindspore/mindformers/blob/dev/configs/README.md)

## API组件注册机制

### 注册原理图

![输入图片说明](https://foruda.gitee.com/images/1673344591842027306/a4d581a3_9324149.png "image-20230110151323393.png")

### 注册机制简介

MindFormers套件全仓代码主要基于Python语言和MindSpore AI框架开发，通过设计业界主流的装饰器注册机制和MindSpore AI框架易用性工具，将套件中涉及的API自动注册至registry字典中，从而实现对API接口的灵活调用。

注册代码设计主要包含两大类： 1. MindFormerModuleType 全局注册模块定义类 2. MindFormerRegister  API注册和实例化类。

通过该注册机制的设定，可以有效的统一套件中各类API接口的调用机制和开发范式，实现套件模块化设计和灵活配置。

### MindFormerModuleType

MindFormerModuleType类目前共支持全局注册模块共22个，可根据实际模型或者工程需求进行增加或改动，详情请见下表：

| MindFormerModuleType |                    |                                                  |
| -------------------- | ------------------ | ------------------------------------------------ |
| 模块名称             | 模块赋值（String） | 模块定义                                         |
| TRAINER              | 'trainer'          | 任务流程创建模块，定义训练/微调/评估/推理流程    |
| PIPELINE             | 'pipeline'         | 推理模块，定义任务的推理流程                     |
| PROCESSOR            | 'procrssor'        | 处理流模块，定义模型权重预处理和加载流程         |
| DATASET              | 'dataset'          | 数据流模块，定义数据集加载及预处理流程           |
| MASK_POLICY          | 'mask_policy'      | 掩码模块，定义掩码策略                           |
| DATASET_LOADER       | 'dataset_loader'   | 数据集加载模块，定义数据集加载策略               |
| DATASET_SAMPLER      | 'dataset_sampler'  | 数据集采样模块，定义数据集采样策略               |
| TRANSFORMS           | 'transforms'       | 数据集增强模块，定义数据集增强策略               |
| MODELS               | 'models'           | 模型网络模块，定义整体网络架构                   |
| ENCODER              | 'encoder'          | 模型骨干网络模块，定义网络中encoder模块          |
| HEAD                 | 'head'             | 模型head模块，定义网络中head头模块               |
| MODULES              | 'modules'          | 模型组件模块，定义网络组件                       |
| BASE_LAYER           | 'base_layer'       | 定义模型组件最小单元模块                         |
| CORE                 | 'core'             | 定义模型组件最小插件模块                         |
| LOSS                 | 'loss'             | 损失函数模块，定义损失函数类型                   |
| LR                   | 'lr'               | 学习策略模块，定义学习策略类型                   |
| OPTIMIZER            | 'optimizer'        | 优化器模块，定义优化器类型                       |
| CONFIG               | 'config'           | 配置模块，定义模型相关的配置                     |
| WRAPPER              | 'wrapper'          | 模型训练封装模块，定义模型前向计算和梯度更新流程 |
| METRIC               | 'metric'           | 模型度量模块，定义模型评估策略                   |
| CALLBACK             | 'callback'         | 回调函数模块，定义模型回调类型                   |
| TOOLS                | 'tools'            | 通用工具模块，定义模型公用工具                   |

### MindFormerRegister

MindFormerRegister 类涉及7类属性函数，主要功能包含：1.  API 注册  2.  API实例化  3.  API检查。详情请见下表：

| 属性函数                 | 入参说明                                                     | 功能说明                     |
| ------------------------ | ------------------------------------------------------------ | ---------------------------- |
| registry                 | dict类型                                                     | 存储已注册的API              |
| register                 | 1. **module_type**(string): MindFormerModuleType包含的模块名，默认’tools’  2. **alias**(string): 注册API重命名，默认None | 装饰器函数，完成API 注册     |
| register_cls             | 1. **register_class**(class or function):  注册API  2. **module_type**(string): MindFormerModuleType包含的模块名，默认’tools’  3. **alias**(string): 注册API重命名，默认None | 注册函数，完成API注册        |
| get_instance_from_config | 1. **config**(dict): 配置文件字典  2. **module_type**(string): MindFormerModuleType包含的模块名，默认’tools’  3. **default_args**(dict):  API默认参数，默认None | 通过配置文件获取API实例      |
| get_instance             | 1. **module_type**(string): MindFormerModuleType包含的模块名，默认’tools’  2. **class_name**(string):  已注册的API名称，默认None  3. ***args**:  定省参数，已注册API支持参数  4. **kwargs: 不定省参数，已注册API支持参数 | 通过已注册API名称获取API实例 |
| get_cls                  | 1. **module_type**(string): MindFormerModuleType包含的模块名，默认’tools’  2. **class_name**(string):  已注册的API名称，默认None | 获取已注册的API类            |
| is_exist                 | 1. **module_type**(string): MindFormerModuleType包含的模块名，默认’common’2. class_name(string):  已注册的API名称，默认None | 检查API是否注册              |

### 注册机制使用示例

- API注册Python代码示例：用户可以利用以下两种注册方式将任意API注册至registry字典中。

```python
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

# Type1 装饰器注册自定义API接口
# 此处即可通过装饰MindFormerRegister的register属性完成 YourDefineLRAPI 类的注册，并将其注册到LR模块
@MindFormerRegister.register(MindFormerModuleType.LR)
class YourDefineLRAPI():
  def __init__(self, *args, **kwargs):
    Pass

# Type2 通过API类进行注册
# 此处即可通过装饰MindFormerRegister的register_cls属性完成 YourDefineAPILR类的注册，并将其注册到LR模块
MindFormerRegister.register_cls(YourDefineLRAPI, MindFormerModuleType.LR)
```

## API 组件 Build 机制

MindFormers套件在注册机制的基础之上，为每个模块都配备的对应的build函数，方便用户使用统一且通用的build形式来创建不同模块的API实例。然而register和build的形式仍然具有一定的`黑盒`特性，需要大家提前了解MindFormers套件中的注册和API创建原理。

### build 支持列表

|     Build Module     |                        功能                        |
| :------------------: | :------------------------------------------------: |
|  [build_callback]()  |   创建回调函数，返回API实例或者包含API实例的列表   |
|     [build_lr]()     |            创建学习率函数，返回API实例             |
|   [build_metric]()   |           创建评估指标函数，返回API实例            |
|   [build_optim]()    |            创建优化器函数，返回API实例             |
|    [build_loss]()    |             创建loss函数，返回API实例              |
|  [build_dataset]()   |       创建数据集，返回MindSpore Dataset实例        |
| [build_dataloader]() |          创建数据集加载函数，返回API实例           |
|  [build_sampler]()   |          创建数据集采样函数，返回API实例           |
| [build_transforms]() | 创建数据增广函数，返回API实例或者包含API实例的列表 |
|    [build_mask]()    |             创建掩码策略，返回API实例              |
|   [build_config]()   |           创建模型配置函数，返回API实例            |
|   [build_model]()    |           创建模型网络函数，返回API实例            |
| [build_processor]()  |    创建数据预处理函数（用于推理），返回API实例     |
| [build_tokenizer]()  |        创建文本的tokenizer函数，返回API实例        |
|   [build_module]()   |           创建网络组件函数，返回API实例            |
|   [build_layer]()    |         创建网络基本构成函数，返回API实例          |
|  [build_pipeline]()  |         创建pipeline推理函数，返回API实例          |
|  [build_trainer]()   |            创建trainer函数，返回API实例            |
|  [build_wrapper]()   |          创建前反向封装函数，返回API实例           |

可以看到，MindFormers套件中目前已经支持20个模块的build，基本覆盖组网过程的各个模块的创建。值得注意的是，我们build函数的设计主要依赖于配置文件（即包含相关模块和相应配置的一个字典文件），因此要充分理解build机制，需要同时学习如何开发配置文件。

### Build 使用案例

下面为用户以某数据集模块的注册和创建使用为例，详细介绍如何使用我们的Build机制来完成一个数据流的创建。

#### 创建分类任务数据集

```text
Note:
    1. MindFormers套件中已经集成的Dataset类均需要依赖配置文件来进行创建，因此使用前可以参考相关配置文件用例来自定义数据任务配置
    2. 如果不想使用配置文件来调用Dataset，用户可以自行定义完整的数据集模块并注册，只需保证返回的实例是mindspore的DATASET类型
```

- Step 1: 开发配置文件 [配置文件链接](https://gitee.com/mindspore/mindformers/blob/dev/configs/mae/run_mae_vit_base_p16_224_800ep.yaml)

```yaml
# train dataset
train_dataset: &train_dataset
  seed: 2022
  batch_size: 64
  data_loader:
    type: ImageFolderDataset
    dataset_dir: "imageNet-1k/train"
    num_parallel_workers: 8
    shuffle: True
  transforms:
    - type: RandomCropDecodeResize
      size: 224
      scale: [0.2, 1.0]
      interpolation: cubic
    - type: RandomHorizontalFlip
      prob: 0.5
    - type: Normalize
      mean: [123.675, 118.575, 103.53]
      std: [58.395, 62.22, 57.375]
    - type: HWC2CHW
  mask_policy:
    type: MaeMask
    input_size: 224
    patch_size: 16
    mask_ratio: 0.75
  input_columns: ["image"]
  output_columns: ["image", "mask", "ids_restore", "unmask_index"]
  column_order: ["image", "mask", "ids_restore", "unmask_index"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  repeat: 1
  numa_enable: False
  prefetch_size: 30
train_dataset_task:
  type: MIMDataset
  dataset_config: *train_dataset
```

- 开发MIMDataset类并注册 [MIMDataset类](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/dataset/mim_dataset.py)

  Note: 如果用户不想依赖配置文件创建数据任务，这里可以由用户自行填写__new__中的创建过程，并保证最后返回的是mindspore dataset实例即可

```python
"""Masked Image Modeling Dataset."""
import os

import mindspore
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.dataset.mask import build_mask
from mindformers.dataset.transforms import build_transforms
from mindformers.dataset.sampler import build_sampler
from mindformers.dataset.base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MIMDataset(BaseDataset):
    """
    Masked Image Modeling Dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import MIMDataset
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['masked_image_modeling']['mae_vit_base_p16']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
            Note:
                The detailed data setting could refer to
                https://gitee.com/mindspore/mindformers/blob/r0.3/docs/model_cards/mae.md
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='MIMDataset',
        ...                                   dataset_config=config.train_dataset_task.dataset_config)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = MIMDataset(config.train_dataset_task.dataset_config)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Masked Image Modeling Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = get_real_rank()
        device_num = get_real_group_size()

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})
        transforms = build_transforms(dataset_config.transforms)
        mask = build_mask(dataset_config.mask_policy)
        sampler = build_sampler(dataset_config.sampler)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            for column in dataset_config.input_columns:
                dataset = get_dataset_map(dataset,
                    input_columns=column,
                    operations=transforms,
                    num_parallel_workers=dataset_config.num_parallel_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing)

        if mask is not None:
            dataset = get_dataset_map(dataset,
                operations=mask,
                input_columns=dataset_config.input_columns,
                output_columns=dataset_config.output_columns,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.project(columns=dataset_config.output_columns)
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
```

- [Build接口](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/dataset/build_dataset.py)：该接口无需用户开发，以及集成在MindFormers套件中供调用

```python
"""Build Dataset API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_dataset(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset', class_name: str = None, **kwargs):
    r"""Build dataset For MindFormer.
    Instantiate the dataset from MindFormerRegister's registry.

    Args:
        config (dict): The task dataset's config. Default: None.
        default_args (dict): The default argument of dataset API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'dataset'.
        class_name (str): The class name of dataset API. Default: None.

    Return:
        The function instance of dataset API.

    Examples:
        >>> from mindformers import build_dataset
        >>> from mindformers.dataset import check_dataset_config
        >>> from mindformers.tools.register import MindFormerConfig
        >>> config = MindFormerConfig('configs/vit/run_vit_base_p16_224_100ep.yaml')
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_class_name = build_dataset(class_name='ImageCLSDataset', dataset_config=config.train_dataset_task)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
```

- 创建Dataset实例

通过build_dataset接口，根据dataset的配置文件来创建MIMDataset实例：

```python
from mindformers.tools.register import MindFormerConfig
from mindformers.dataset import build_dataset, check_dataset_config
# Initialize a MindFormerConfig instance with a specific config file of yaml.
config = MindFormerConfig("configs/mae/task_config/mim_dataset.yaml")
check_dataset_config(config)

# 1) use config dict to build dataset
dataset_from_config = build_dataset(config.train_dataset_task) # 通过train_dataset_task关键字来创建是MIMDataset API实例
# 2) use class name to build dataset
dataset_from_name = build_dataset(class_name='MIMDataset', dataset_config=config.train_dataset_task)
# 3) use class to build dataset
dataset_from_class = MIMDataset(config.train_dataset_task)
# 4) 不使用配置文件创建dataset实例
dataset_from_self_define1 = build_dataset(class_name='MIMDataset')
dataset_from_self_define2 = MIMDataset()
```
