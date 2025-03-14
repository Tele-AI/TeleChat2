# PromptFlashAttention(PFA)和IncreFlashAttention(IFA)接入指南

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.5.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 概述

PromptFlashAttention(PFA)
在算法中可以取代Self-Attention的计算，目前在算法中可以获得性能以及显存收益。PromptFlashAttention仅可用于全量推理，目前不支持增量推理场景(
seq_length=1)且不可用于训练。PFA支持多卡场景。

IncreFlashAttention(IFA)仅支持增量推理场景下的非首次推理(seq_length=1)
，且不可用于训练。IFA目前不支持多卡场景。因此目前在GPT2中的分布式增量推理的场景为PFA + SA，而单卡推理的场景才为PFA + IFA。

## API介绍

### PromptFlashAttention

```python
class PromptFlashAttention(Primitive):
    r"""
    The interface for fully inference.
    B -- Batch size
    S -- Sequence length
    H -- Hidden size

    Note:
    experiment ops

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        num_heads (int): The number of heads.
        scale_value (float): The scale value indicating the scale coefficient, which is used as the scalar of
          Muls in the calculation. Default: 1.0.
        pre_tokens (int): Previous tokens. Default: 2147483547.
        next_tokens (int): next tokens.  Default: 0.
          indicate the upper triangle, Indicate the number of data blocks involved in the calculation. The value 0
          indicates that the data blocks in the upper triangle are not involved in the calculation
        input_layout (str): the data layout of the input qkv, support `(BSH)` and `(BNSD)`, Default `BSH`.
        num_key_value_heads (int): head numbers of key/value which are used in GQA algorithm.
          The value o indicates if the key and value have the same head nums, use numHeads.  Default: 0.
        sparse_mode (int): Default: 0
        inner_precise (int): 0, float16 high precision. 1, high performance. default 1

    Inputs:
        - **query** (Tensor) - The query tensor with data type of float16 or float32.
          Input tensor of shape :math:`(B, S, H)` / `(B, N, S, D)`.
        - **key** (Tensor) - The key tensor with data type of float16 or float32.
          Input tensor of shape :math:`(B, S, H)` / `(B, N, S, D)`.
        - **value** (Tensor) - The value tensor with data type of float16 or float32.
          Input tensor of shape :math:`(B, S, H)` / `(B, N, S, D)`.
        - **attn_mask** (Tensor) - The attention mask tensor with data type of float16 or float32.
          For each element, 0 indicates retention and 1 indicates discard. Input tensor of shape :math:`(B, 1, S, S)`.
        - **padding_mask** (Tensor) - The padding mask tensor with data type of float16 or float32
        - **actual_seq_lengths** (Tensor): Describe actual sequence length of each input with data type of int64.
        - **actual_seq_lengths_kv** (Tensor): Describe actual sequence length of each input with data type of int64.
        - **dep_scale1** (Tensor)
        - **quant_scale1** (Tensor)
        - **deq_scale2** (Tensor)
        - **quant_scale2** (Tensor)
        - **quant_offset2** (Tensor)

    Outputs:
        - **attention_out** (Tensor) - Input tensor of shape :math:`(B, S, H)` / `(B, N, S, D)`.
    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore.ops.operations.nn_ops as P
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> B = 1
        >>> N = 16
        >>> S = 256
        >>> D = 16
        >>> query = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        >>> key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        >>> value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        >>> attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
        >>> pfa = P.PromptFlashAttention(N, input_layout='BNSD')
        >>> out = pfa(query, key, value, attn_mask, None, None, None, None, None, None, None, None)
        >>> print(out[0].shape)
        (1, 16, 256, 16)
    """
```

其中pre_token和next_token的意义为将一个`attention_mask`的左上角向右偏移`next_tokens`
个位置，从这个位置向右下45°画一条线；右下角向左偏移`pre_tokens`
个位置，向左上45°画一条线。这两条线相交的位置为有效的`attention_mask`。其他的入参意义比较好理解，见上述API文档。

### IncreFlashAttention

```python
class IncreFlashAttention(Primitive):
    r"""
    The interface for fully inference.

    B -- Batch size

    S -- Sequence length

    H -- Hidden size

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **query** (Tensor) - The query tensor with data type of float16 or bfloat16.
          Input tensor of shape :math:`(B, 1, H)` / :math:`(B, N, 1, D)`.
        - **key** (TensorList) - The key tensor with data type of float16 or bfloat16.
          Input tensor of shape :math:`(B, S, H)` / :math:`(B, N, S, D)`.
        - **value** (TensorList) - The value tensor with data type of float16 or bfloat16.
          Input tensor of shape :math:`(B, S, H)` / :math:`(B, N, S, D)`.
        - **attn_mask** (Tensor) - The attention mask tensor with data type of float16 or bool.
          Input tensor of shape :math:`(B, S)` / :math:`(B, 1, S)` / :math:`(B, 1, 1, S)`.
        - **actual_seq_lengths** (Tensor) - Describe actual sequence length of each input with data type of int.
        - **padding_mask** (Tensor) - The padding mask tensor with data type of float16.
        - **dequant_scale1** (Tensor) - Quantitative parametor, the tensor with data type of uint64.
        - **quant_scale1** (Tensor) - Quantitative parametor, the tensor with data type of float.
        - **dequant_scale2** (Tensor) - Quantitative parametor, the tensor with data type of uint64.
        - **quant_scale2** (Tensor) - Quantitative parametor, the tensor with data type of float.
        - **quant_offset2** (Tensor) - Quantitative parametor, the tensor with data type of float.
        - **antiquant_scale** (Tensor) - Quantitative parametor, the tensor with data type of float.
        - **antiquant_offset** (Tensor) - Quantitative parametor, the tensor with data type of float.
        - **block_table** (Tensor) - The tensor with data type of float.
        - **num_heads**  (int) - The number of heads.
        - **input_layout** (str) - the data layout of the input qkv, support `(BSH)` and `(BNSD)`. Default `BSH`.
        - **scale_value** (double) - The scale value indicating the scale coefficient, which is used as the scalar of
          Muls in the calculation. Default: 1.0.
        - **num_key_value_heads** (int) - head numbers of key/value which are used in GQA algorithm.
          The value o indicates if the key and value have the same head nums, use numHeads.  Default: 0.
        - **block_size** (int) - Default: 0.
        - **inner_precise** (int) - Default: 1.

    Outputs:
        - **attention_out** (Tensor) - Input tensor of shape :math:`(B, 1, H)` / :math:`(B, N, 1, D)`.

    Supported Platforms:
        ``Ascend``
    """
```

IFA的入参和PFA的入参基本一致，参考上述API文档，但注意IFA为仅支持非首次推理的增量图里场景，所以`key`的`seq_length`为1。

## 使用方法

在GPT2中PFA的定义和使用如下：

```python
self.prompt_flash_attention = PromptFlashAttention(num_heads=num_heads,
                                                   scale_value=1.0,
                                                   pre_tokens=self.src_seq_length,
                                                   next_tokens=0,
                                                   input_layout='BNSD',
                                                   num_key_value_heads=0)
attention = self.prompt_flash_attention(query, key, value, attention_mask,
                                        None, None, None, None, None, None, None, None)[0]
```

在GPT2中IFA的定义以及使用如下：

```python
self.incre_flash_attention = IncreFlashAttention(num_heads=num_heads,
                                                 scale_value=1.0,
                                                 input_layout='BNSD',
                                                 num_key_value_heads=0)
attention = self.incre_flash_attention(query, key, value, attention_mask,
                                       None, None, None, None, None, None, None, None)
```

IFA的输入目前可以参考PFA，但是`query`和`value`在文档中支持的是TensorList，可能与当前用例不太一致，接口可能会有调整。请以最新的文档为准。

## 代码修改

PFA和IFA的调用可以直接替换原有的`_attn`的逻辑即可。有以下几点需要修改：

1. attn中的merge_head在PFA和IFA中没有，需要单独调用。
2. 如果设置scale_value为1.0时，那么手动将输入进行normalize(除以sqrt(d))，否则计算将会不等价。
3. attention_mask的翻转逻辑原来在attn中计算，现在需要将翻转完的attention_mask作为入参传入PFA和IFA中。

```python
if not self.training and self.use_prompt_flash_attention:
    if self.use_past and not self.is_first_iteration:
        if self.use_incre_flash_attention:
            query, key, attention_mask = self._pfa_ifa_data_preprocess(query, key, attention_mask,
                                                                       batch_valid_length)
            attention = self.incre_flash_attention(query, key, value, attention_mask,
                                                   None, None, None, None, None, None, None, None)
            attention = self._merge_heads(attention)
        else:
            key = self.transpose(key, (0, 1, 3, 2))
            attention = self._attn(query, key, value, attention_mask, batch_valid_length)
    else:
        query, key, attention_mask = self._pfa_ifa_data_preprocess(query, key, attention_mask,
                                                                   batch_valid_length)
        attention = self.prompt_flash_attention(query, key, value, attention_mask,
                                                None, None, None, None, None, None, None, None)[0]
        attention = self._merge_heads(attention)
elif self.use_flash_attention:
    attention = self._flash_attn(query, key, value, attention_mask)
else:
    attention = self._attn(query, key, value, attention_mask, batch_valid_length)
```
