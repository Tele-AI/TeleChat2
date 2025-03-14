# ConvertWeight

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 基本介绍

ConvertWeight支持对torch权重和mindspore权重的格式互转

## 支持模型

| name        |                          |
|-------------|--------------------------|
| llama       | llama                    |
| glm         | glm                      |
| glm-n       | glm2、glm3                |
| qwen        | qwen、qwen2              |
| internlm    | internlm                 |
| baichuan    | baichuan、baichuan2       |
| gpt         | gpt2                     |
| wizardcoder | wizardcoder              |
| skywork     | skywork                  |
| mixtral     | mixtral                  |
| mae         | mae                      |
| vit         | vit                      |
| swin        | swin                     |
| knowlm      | knowlm                   |
| telechat    | telechat_7b、telechat_12b |
| codegeex2   | codegeex2                |
| deepseek    | deepseek、deepseek1_5     |

## 使用方式

### 启动权重转换脚本

脚本：

根目录下`convert_weight.py`

主要参数;

| args          |                                                      | required |
|---------------|------------------------------------------------------|----------|
| model         | 模型名称                                                 | 是        |
| reversed      | mindspore格式转torch格式                                  | 否        |
| input_path    | 输入权重文件路径，如果模型存在多个权重文件，选择模型目录下其中一个即可，根据目录自动加载全部权重     | 是        |
| output_path   | 输出权重文件路径                                             | 是        |
| dtype         | 输出的权重数据类型,默认为原始权重数据类型                                | 否        |
| layers        | gpt2和wizardcoder的torch权重转mindspore权重时所需的额外参数，转换的权重层数 | 否        |
| is_pretrain   | swin权重转换所需额外参数，输入权重是否为预训练权重                          | 否        |
| telechat_type | telechat权重转换所需额外参数，模型版本                              | 否        |

执行：

```shell
python convert_weight.py --model model_name --input_path ./hf/input.bin --output_path ./ms/output.ckpt --otherargs
python convert_weight.py --model model_name --input_path ./ms/output.ckpt --output_path ./hf/input.bin --reversed --otherargs
# Example for llama:
# python convert_weight.py --model llama --input_path open_llama_7b.ckpt --output_path llama_7b.bin --reversed

```

## 扩展

1. 在扩展模型目录下新增`convert_weight.py`及`convert_reversed.py`文件，
2. 在文件中分别编写conver_ms_to_pt及conver_pt_to_ms权重转换函数，函数参数为`input_path`、`output_path`、`dtype`及额外参数`**kwargs`
3. 在mindformers根目录下`convert_weight.py`文件中的convert_map和reversed_convert_map字典中加入扩展模型名称及转换函数引入路径
4. 额外参数在main函数中通过`parser.add_argument('--arg_name',default=,type=,required=,help=)`新增
