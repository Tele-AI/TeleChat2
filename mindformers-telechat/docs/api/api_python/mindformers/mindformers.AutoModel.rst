mindformers.AutoModel
========================

.. py:class:: mindformers.AutoModel(*args, **kwargs)

    这是一个通用的模型类，使用类方法 `AutoModel.from_pretrained` 或 `AutoModel.from_config` 创建时会自动实例化为库中的基础模型类之一。
    这个类不能直接使用 \_\_init\_\_() 实例化（会抛出异常）。

    .. py:method:: from_config(config, **kwargs)
        :classmethod:

        通过Config实例或者yaml文件实例化一个库中的基础模型类。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        .. note::
            通过其配置文件加载模型 **不会** 载入模型权重。这只影响模型的配置。使用 `AutoModel.from_pretrained` 以载入模型权重。

        参数：
            - **config** (Union[MindFormerConfig, PretrainedConfig, str]) - `MindFormerConfig` 实例，yaml文件路径，或者 `PretrainedConfig` 实例（实验特性）。实例化得到的模型类将基于以下配置类进行选择：

              - `BertConfig` 配置类： `BertModel` ( `BertModel` 模型)
              - `Blip2Config` 配置类： `Blip2Model` ( `Blip2Llm` 模型)
              - `BloomConfig` 配置类： `BloomModel` ( `BloomModel` 模型)
              - `CLIPConfig` 配置类： `CLIPModel` ( `CLIPModel` 模型)
              - `ChatGLM2Config` 配置类： `ChatGLM2Model` ( `ChatGLM2Model` 模型)
              - `GLMConfig` 配置类： `GLMModel` ( `GLMChatModel` 模型)
              - `GPT2Config` 配置类： `GPT2Model` ( `GPT2Model` 模型)
              - `LlamaConfig` 配置类： `LlamaModel` ( `LlamaModel` 模型)
              - `PanguAlphaConfig` 配置类： `PanguAlphaModel` ( `PanguAlphaModel` 模型)
              - `SamConfig` 配置类： `SamModel` ( `SamModel` 模型)
              - `SwinConfig` 配置类： `SwinModel` ( `SwinModel` 模型)
              - `T5Config` 配置类： `T5Model` ( `T5ForConditionalGeneration` 模型)
              - `ViTConfig` 配置类： `ViTModel` ( `ViTModel` 模型)
              - `ViTMAEConfig` 配置类： `ViTMAEModel` ( `ViTMAEModel` 模型)

            - **kwargs** (Dict[str, Any], 可选) - 传入的配置信息将会覆盖config中的配置信息。

        返回：
            一个模型实例。

    .. py:method:: from_pretrained(pretrained_model_name_or_dir, *model_args, **kwargs)
        :classmethod:

        从文件夹、或魔乐社区读取配置信息，实例化一个库中的基础模型类。

        实例化的模型会基于配置对象（通过参数传入，或在存在的情况下从目录 `pretrained_model_name_or_dir` 中载入）的 `model_type` 属性选择模型类别。若配置对象缺失，则会对 `pretrained_model_name_or_dir` 进行模式匹配：

        - **bert** - `BertModel` ( `BertModel` 模型)
        - **blip2** - `Blip2Model` ( `Blip2Llm` 模型)
        - **bloom** - `BloomModel` ( `BloomModel` 模型)
        - **clip** - `CLIPModel` ( `CLIPModel` 模型)
        - **glm** - `GLMModel` ( `GLMChatModel` 模型)
        - **glm2** - `ChatGLM2Model` ( `ChatGLM2Model` 模型)
        - **gpt2** - `GPT2Model` ( `GPT2Model` 模型)
        - **llama** - `LlamaModel` ( `LlamaModel` 模型)
        - **mae** - `ViTMAEModel` ( `ViTMAEModel` 模型)
        - **pangualpha** - `PanguAlphaModel` ( `PanguAlphaModel` 模型)
        - **sam** - `SamModel` ( `SamModel` 模型)
        - **swin** - `SwinModel` ( `SwinModel` 模型)
        - **t5** - `T5Model` ( `T5ForConditionalGeneration` 模型)
        - **vit** - `ViTModel` ( `ViTModel` 模型)

        模型会默认调用 `model.eval()` 以设置为评估模式（例如dropout模块被停用）。要训练模型，您应该首先使用 `model.train()` 将其设置回训练模式。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **pretrained_model_name_or_dir** (str) - 包含yaml文件和ckpt文件的文件夹路径、包含config.json文件和对应的权重文件的文件夹路径、或魔乐社区上的model_id。后两者为实验特性。
            - **model_args** (Any, 可选) - 会在实例化模型时，传给模型的 \_\_init\_\_() 方法。仅在实验特性时生效。
            - **kwargs** (Dict[str, Any], 可选) - 可用于更新载入的配置对象和实例化模型（例如 `output_attentions=True` ）。
              当 `config` 已通过参数提供或者已自动载入时， `**kwargs` 会被传入模型的 `__init__` 方法；否则 `**kwargs` 会首先传入 `PretrainedConfig.from_pretrained` 方法构建一个配置对象，而与配置属性无关的键则会传入模型的 `__init__` 方法。

              部分可用的键如下所示:

              - config (PretrainedConfig, 可选):
                模型的配置信息，用于替换自动载入的配置。默认值： ``None`` 。
                在以下情况下会自动载入模型配置：

                - 模型通过预训练模型的model_id从库中载入。
                - 模型由 `PreTrainedModel.save_pretrained` 方法保存并通过提供保存目录载入。
                - 模型由 `pretrained_model_name_or_dir` 表示的本地目录载入，目录中包含一个命名为 'config.json' 的配置文件。

              - cache_dir (Union[str, os.PathLike], 可选):
                在标准缓存不可用的情况下，一个缓存了下载的预训练模型配置文件的文件夹路径。默认值： ``None`` 。
              - force_download (bool, 可选):
                是否强制（重新）下载模型权重和配置文件，这将覆盖掉原本缓存的版本。默认值： ``False`` 。
              - resume_download (bool, 可选):
                是否删除未完全接收的文件。将会在此类文件存在时尝试恢复下载。默认值： ``False`` 。
              - proxies (Dict[str, str], 可选):
                一个协议或端点使用的代理服务器字典，例如 `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}` 。代理会在每一次请求中使用。默认值： ``None`` 。
              - local_files_only (bool, 可选):
                是否只查看本地文件（不尝试下载模型）。默认值： ``False`` 。
              - revision (str, 可选):
                使用特定的模型版本。可以是一个分支名、标签名、一个提交，或者任何git允许的标识符。默认值： ``"main"`` 。
              - trust_remote_code (bool, 可选):
                是否允许在自己的建模文件中对Hub上定义的模型进行自定义。该选项应当仅对你信任且阅读过代码的仓库设置为 ``True`` ，因为这会在你的本地机器上执行Hub当前的代码。默认值： ``False`` 。
              - code_revision (str, 可选):
                当代码与模型的其他部分位于不同的仓库时，使用Hub上代码的特定修订。可以是一个分支名、标签名、一个提交，或者任何git允许的标识符。默认值： ``"main"`` 。

        返回：
            一个继承自PretrainedModel类的模型实例。

    .. py:method:: register(config_class, model_class, exist_ok=False)
        :classmethod:

        注册一个新的模型类到此类中。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **config_class** (PretrainedConfig) - 模型的Config类。
            - **model_class** (PretrainedModel) - 用于注册的模型类。
            - **exist_ok** (bool, 可选) - 为True时，即使 `config_class` 已存在也不会报错。默认值： ``False`` 。
