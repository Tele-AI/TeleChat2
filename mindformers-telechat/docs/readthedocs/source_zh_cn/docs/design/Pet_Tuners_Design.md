# 低参微调

当前基于Transformer结构的LLMs取得了显著的成就，如何更好地利用已经开源的语言预训练大模型来应用到自己的应用场景中，往往需要使用微调来达到较好的。微调又分为两种：全参微调和低参微调，使用全参微调需要较大的算力，这往往对于普通开发者来说很难获取到足够大的算力资源来进行自己的实验；此时就需要使用低参微调（Parameters Efficient Tuning），只需要训练极少量的参数就可以接近全参微调的效果，以此突破算力限制。

当前，MindFormer是专注于Transformer结构的大模型的开发，对外使能开发者开发大模型，训练大模型；MindPet则是专注于低参微调算法的开发，对外提供低参微调算法接口，使能开发者修改自己开发的模型进行低参微调。但是，以上两个工具包均是使能各自的开发，而对于完整的大模型开发，则需要大模型的开发->大模型训练->低参微调的流程链路，当前存在断点，则希望在本工程中将MindFormer的能力进行拓展，能支持MindFormer开发的大模型自动的链接上MindPet上的低参微调算法，加速大模型的孵化。

## 设计概述

由于是两个完全独立的开发组件，需要结合两者的功能来完成对于大模型的微调，此时使用适配器设计思路去整合两个部件则是最合适的。由此整体的设计如下图：

![整体设计图](assets/MindPet.png)

在本部分则是实现上图绿色部分的功能，调用微调组件的APIs，针对用户输入的模型进行适配修改，然后返回给用户，用户可以将其送入到MindFormer的Trainer接口中，进行后续的模型微调训练。

## 接口设计

**PetAdapter**

```python
class PetAdapter:
    r"""
    PetAdapter is the base class of adapter to modify the pretrained model.
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        """Add efficient tuning parameters to ptm."""
        raise(NotImplementedError("should implemented by the certain tuning algorithm."))

    @classmethod
    def freeze_pretrained_model(cls, model, pet_type:PetType):
        pass
```

PetAdapter是所有微调算法适配器的基类，针对于具体的微调算法，一般只需要实现实现`get_pet_model()`接口，将微调算法的改动添加到用户的模型中。

**PetAdapter.get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None)**

``` text
功能:

获取微调算法适配的预训练模型

参数:

- model(nn.Cell)：原预训练模型，支持用户直接输入模型实例，默认是None，当输入是None值，则是从config中由用户输入的模型实例化MindFormer中支持的模型。
- config(PetConfig)：微调算法的配置，包含微调算法的超参或者需要实例化的预训练模型。

返回值：

适配了微调算法的预训练模型(nn.Cell)
```

**PetAdapter.freeze_pretrained_model(cls, model, pet_type:PetType)**

``` text
功能:

根据微调算法类型冻结预训练模型权重

参数:

- model：需要冻结权重的预训练模型
- pet_type： 微调算法类型
```
