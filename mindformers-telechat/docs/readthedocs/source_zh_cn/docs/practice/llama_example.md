# LLaMA 迁移指南

## 1 改造GPT2模型及LOSS脚本

### 1.1 熟悉模型架构

- 阅读LLaMA论文[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)，明确LLaMA是Transformers Decoder结构，具有以下特点：

1. 使用RMSNorm作为归一化层；
2. 使用RotaryEmbedding进行相对位置编码；
3. FeedForward中使用门控结构，激活函数为SiLU；
4. 没有Dropout层，线性层中没有bias。

- 查看[META官方代码](https://github.com/facebookresearch/llama/blob/main/llama/model.py)或[Huggingface代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)，明确论文实现细节，基于GPT2改造`Mindformers`中`RMSNorm`, `RotaryEmbedding`, `FeedForward`模块，以及修改`Attention`与`Decoder`模块。
- 注意META官方代码与Huggingface代码两者`RotaryEmbedding`模块实现逻辑不同，权重文件也有两种，需要注意区分，这里我们使用Huggingface的实现方式。

  ```txt
  Mindformers
    ├── scripts
    │   └── run_distribute.sh
    └── mindformers
        ├── model
        └── model
            └── llama
                ├── llama_layer.py
                ├── llama_transformer.py
                └── llama.py
  ```

### 1.2 RMSNorm 实现

Llama使用RMSNorm代替传统LayerNorm，降低计算复杂度。
具体实现代码参考[https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py)。
实现RMSNorm的代码：

```Python
class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.
    """
    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim,), dtype=mstype.float32), parallel_optimizer=False)
        self.square = P.Square()
        self.mean = P.ReduceMean(keep_dims=True)
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.compute_type = compute_type

    def _norm(self, x):
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        x = self.cast(x, mstype.float16)
        norm_factor = self.cast(norm_factor, mstype.float16)
        return self.mul(x, norm_factor)

    def construct(self, x):
        """Forward of RMSNorm."""
        original_type = x.dtype
        x = self.cast(x, self.compute_type)
        output = self._norm(x)
        output = self.cast(output, mstype.float16)
        weight = self.cast(self.weight, mstype.float16)
        output = self.mul2(output, weight)
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy_in):
        """Parallel strategy configuratiuon interface."""
        self.square.shard((strategy_in,))
        self.mean.shard((strategy_in,))
        self.rsqrt.shard((strategy_in,))
        self.add.shard((strategy_in, ()))
        self.mul.shard((strategy_in, strategy_in))
        self.mul2.shard((strategy_in, (1,)))

```

### 1.3 RotaryEmbedding 实现

Llama的位置编码在Attention中进行，使用RotaryEmbedding实现相对位置编码。
具体实现代码参考[https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py)。
实现RotaryEmbedding的代码如下：

```Python
def get_swap_mask(head_dim):
    """Swap matrix"""
    zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
    id_block = np.identity(head_dim // 2, dtype=np.float32)
    return np.block([[zero_block, id_block], [-id_block, zero_block]])


def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        dtype=mstype.float32,
        pretrain_seqlen=2048,
        extend_method=SeqExtendMethod.NONE.value):
    """
    Precompute of freqs and mask for rotary embedding.
    """
    ratio = 1.
    if extend_method != SeqExtendMethod.NONE.value and end > pretrain_seqlen:
        ratio = end / pretrain_seqlen
    if extend_method == SeqExtendMethod.NTK.value:
        theta *= ratio
    freqs_base = np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) # (head_dim // 2, )
    freqs = 1.0 / (theta ** (freqs_base / dim)) # (head_dim // 2, )
    if extend_method == SeqExtendMethod.PI.value:
        t = np.arange(0, end / ratio, 1 / ratio).astype(np.float32)
    else:
        t = np.arange(0, end, 1).astype(np.float32)  # type: ignore # (seq_len,)
    freqs = np.outer(t, freqs)  # type: ignore (seq_len, head_dim // 2)
    emb = np.concatenate((freqs, freqs), axis=-1)
    freqs_cos = np.cos(emb) # (seq_len, head_dim)
    freqs_sin = np.sin(emb) # (seq_len, head_dim)
    freqs_cos = Tensor(freqs_cos, dtype=dtype)
    freqs_sin = Tensor(freqs_sin, dtype=dtype)

    swap_mask = get_swap_mask(dim)
    swap_mask = Tensor(swap_mask, dtype=dtype)

    return freqs_cos, freqs_sin, swap_mask


class LlamaRotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding.
    """

    def __init__(self, head_dim=128):
        super().__init__(auto_prefix=False)
        self.head_dim = head_dim

        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()

    # rotary pos emb helpers:
    def rotate_half(self, x, swap_mask):
        # shard:(dp, mp, 1, 1)
        x = self.bmm_swap(x, swap_mask)
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        # xq, xk: [b, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        xq_out = self.add(self.mul(xq, freqs_cos),
                          self.mul(self.rotate_half(xq, swap_mask), freqs_sin))
        xk_out = self.add(self.mul(xk, freqs_cos),
                          self.mul(self.rotate_half(xk, swap_mask), freqs_sin))
        return xq_out, xk_out

    def shard(self, strategy_in):
        self.add.shard((strategy_in, strategy_in))
        self.bmm_swap.shard((strategy_in, (1, 1)))
        self.mul.shard((strategy_in, (strategy_in[0], 1, 1, 1)))
```

### 1.4 FeedForward

Llama的FeedForward相比于GPT包含一个额外Gate的结构。
具体实现代码参考[https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_layer.py)。
改造FeedForward：

```Python

class LlamaFeedForward(Cell):
    r"""
    LLaMA FeedForward.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 hidden_dim,
                 multiple_of,
                 hidden_act=LlamaSiLU,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32):
        super().__init__()

        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")

        if ffn_dim_multiplier is not None:
            hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         activation=hidden_act,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x) # dp,1 -> dp, mp
        hidden = self.w3(x) # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate) # dp,mp -> dp, mp
        output = self.w2(hidden) # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
        self.w1.activation.shard(((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))

```

### 1.5 改造`Attention`与`Decoder`模块

- 修改`Attention`与`Decoder`模块以适配LLaMA网络中使用的rotaryembedding, feedforward, RMSNorm，并去掉Dropout层与线性层中的bias。
- 具体实现参考具体实现代码参考[https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_transformer.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama_transformer.py)。
- 核心计算逻辑如下：

```Python
class LLamaAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in LLaMA.
    """
    .
    .
    .
    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask_adder=None,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        x = self.reshape(x, (-1, x.shape[-1]))
        # [bs * seq/1, hidden_dim]
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key = self.cast(self.wk(x), self.dtype)    # dp, 1 -> dp, mp
        value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        query = self.reshape(query, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_head, self.head_dim))
        key = self.reshape(key, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                 self.n_kv_head, self.head_dim))
        value = self.reshape(value, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_kv_head, self.head_dim))
        # [bs, seq/1, n_head/n_kv_head, head_dim]
        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))
        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis)
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (
                    self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = (self.equal(self.range, valid_length)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                current_value = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add_past(key_past, current_key)
                value = self.add_past(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        attention = self._attn(query, key, value, mask_adder)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)
        output = self.cast(output, ori_dtype)

        return output, layer_present

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
        x = self.reshape(x, (bs * n_kv_head, 1, seqlen, head_dim))
        x = self.tile_kv(x, (1, rep, 1, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output
        """
        # dp,mp,1,1 -> dp,1,mp,1
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, head_dim
        x_shape = x.shape
        if self.compute_in_2d:
            # [bs * seq/1, hidden_dim]
            new_shape = (-1, x_shape[-2] * x_shape[-1])
        else:
            # [bs, seq/1, hidden_dim]
            new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask_adder):
        """
        Get the weighted score along the seq_length
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask_adder, score)

        attention_probs = self.softmax(self.cast(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge
```

```Python
class LLamaDecodeLayer(nn.Cell):
    r"""
    LLamaDecodeLayer Layer.
    """
    .
    .
    .
    def construct(self, x, freqs_cis, mask_adder=None, init_reset=True, batch_valid_length=None):
        # [bs, seq/1, hidden_dim] (first) [bs * seq/1, hidden_dim] (others)
        if self.compute_in_2d:
            x = self.reshape(x, (-1, x.shape[-1]))
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        input_x = self.attention_norm(x)

        key_reset = None
        value_reset = None
        if self.use_past and self.is_first_iteration:
            # reset states, init_reset True for reuse and False for reset
            self.assign_past(self.key_past, self.mul_past(self.key_past, self.cast(init_reset, self.dtype)))
            self.assign_past(self.value_past, self.mul_past(self.value_past, self.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = ops.depend(input_x, key_reset)
            input_x = ops.depend(input_x, value_reset)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        h, layer_present = self.attention(input_x, freqs_cis, mask_adder,
                                          self.key_past, self.value_past, batch_valid_length)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign_past(self.key_past, key_present)
            self.assign_past(self.value_past, value_present)
            key_update = self.key_past
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = ops.depend(key_update, key_reset)
            value_update = ops.depend(value_update, value_reset)

        # add dependency for desired execution order
        ffn_out = ops.depend(ffn_out, value_update)
        ffn_out = ops.depend(ffn_out, key_update)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out, layer_present
```

### 1.6 实现`LlamaModel`与`LlamaForCausalLM`

- 参照GPT2的`GPT2Model`与`GPT2LMHeadModel`实现`LlamaModel`与`LlamaForCausalLM`。
 - 具体实现代码参考[https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama.py)。
- 核心计算逻辑如下：

```Python
class LlamaModel(PreTrainedModel):
    .
    .
    .
    def construct(self, tokens: Tensor, input_position=None, init_reset=True, batch_valid_length=None):
        """Forward of llama model."""
        # preprocess
        bs, seq_len = tokens.shape
        if self.is_first_iteration:
            freqs_cis = (self.tile(self.reshape(self.freqs_cos, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(self.freqs_sin, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.swap_mask)
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
            mask = self.get_attention_mask(input_mask)
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            freqs_cis = (self.reshape(self.gather_past(self.freqs_cos, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.reshape(self.gather_past(self.freqs_sin, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.swap_mask)
            mask = self.expand_dims(self.cast(self.le_past(self.range, valid_length), self.dtype), 2)
        if mask.ndim == 3:
            mask = self.expand_dims(mask, 1)
        mask = self.sub(self.one, mask.astype(self.dtype))
        mask_adder = self.mul_mask(mask, self.multiply_data)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, freqs_cis, mask_adder,
                                  init_reset=init_reset, batch_valid_length=batch_valid_length)
        output = self.norm_out(h)
        return output
```

```Python
class LlamaForCausalLM(PreTrainedModel):
    .
    .
    .
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """LlamaForCausalLM forward."""
        bsz, seqlen = input_ids.shape
        if self.phase == "train":
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position, init_reset, batch_valid_length)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
            label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
            if input_mask.shape[-1] == label_mask.shape[-1]:
                input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if self.phase != "train":
            logits = self.reshape(logits, (bsz, seqlen, -1))

            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

```

### 1.7 配置并行策略

Mindformers 支持数据并行(dp)、模型并行(mp)、流水线并行(pp)等多种并行策略。
下面介绍如何在半自动并行模式下配置并行切分策略。
在`Attention`模块配置算子切分的shard：

```Python
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.softmax.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp * mp, 1, 1, 1),))

            self.apply_rotary_emb.shard((dp, mp, 1, 1))
            self.wq.shard(((dp, 1), (mp, 1)))
            self.wk.shard(((dp, 1), (mp, 1)))
            self.wv.shard(((dp, 1), (mp, 1)))
            self.wo.shard(((dp, mp), (1, mp)))
```

在`Decoder`模块配置算子切分的shard：

```Python
            self.feed_forward.shard(parallel_config)
            if self.compute_in_2d:
                self.add.shard(((dp, 1), (dp, 1)))
                self.attention_norm.shard((dp, 1))
                self.ffn_norm.shard((dp, 1))
            else:
                self.add.shard(((dp, 1, 1), (dp, 1, 1)))
                self.attention_norm.shard((dp, 1, 1))
                self.ffn_norm.shard((dp, 1, 1))
                self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))
```

在`LlamaModel`模块配置算子切分的shard与pipeline的stage：

```Python
def layer_compute_dtype(layer, layer_id, offset, parallel_config, n_layers):
    pp_dis = max(int((n_layers + 1) / parallel_config.pipeline_stage), 1)
    pp_id = min((layer_id + offset) // pp_dis,
                parallel_config.pipeline_stage - 1)
    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            layer.recompute()
    else:
        if parallel_config.recompute.recompute:
            layer.recompute(
                recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)
```

```Python
        for layer_id in range(config.num_layers):
            layer = LLamaDecodeLayer(...)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, use_select_recompute=config.use_select_recompute)
            self.layers.append(layer)
        .
        .
        .
        self.tok_embeddings.pipeline_stage = 0
        if config.parallel_config.pipeline_stage > 1:
            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            self.tok_embeddings.set_comm_fusion(2)
            self.norm_out.set_comm_fusion(2)
        else:
            self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.tok_embeddings.shard(config.parallel_config)

        self.tile.shard(((1, 1, 1, 1), ()))
        self.sub.shard(((1,), (dp, 1, 1, 1)))
        self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
        self.expand_dims.shard(((dp, 1, 1),))
        self.not_equal.shard(((dp, 1), ()))
        self.gather.shard(((dp, 1), (1,)))
        if config.compute_in_2d:
            self.norm_out.shard((dp, 1))
        else:
            self.norm_out.shard((dp, 1, 1))

```

## 2 修改数据处理脚本

数据处理的方式通常伴随着数据集同时提供。这里以Alpaca数据集为例，分为以下两个步骤：

- 将问答数据按照模板填入。参考[`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py)。
- 将模板化的问答转换成input_tokens和target_tokens并保存成mindrecord。参考[`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py)。

其中第二步，使用了llama专属的tokenizer将模板化的问答转换成input_tokens，而target_tokens则是将input_tokens的非回答部分置成特殊字符ignore token，从而在训练时，loss只对回答部分计算。

## 3 修改Tokenizer

Tokenizer是一个将单词和整数之间映射和反映射的功能模块。一般具体模型都提供对应的映射脚本。以LLAMA为例，LLAMA的tokenizer是基于`sentencepiece`库加载的词表文件。
具体LLAMA tokenizer参考`/mindformers/models/llama/llama_tokenizer.py`
其中涉及到`sentencepiece`库加载的词表文件脚本为：

```python
import sentencepiece as spm
self.vocab_file = vocab_file
self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
self.sp_model.Load(vocab_file)
```

同时在初始化tokenizer时，需要将特殊token明确指明。相关脚本如下：

```python
    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
    super().__init__(
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token=pad_token,
        sp_model_kwargs=self.sp_model_kwargs,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs,
    )
```

## 4 权重转换脚本

权重文件是一个key, value配对的字典，key为权重的名称，value为权重的值。
将HuggingFace的权重文件转换到MindFormers权重文件，涉及到权重名称的改变，也就是将文件中的key按照mindformers的名称规范重命名。

具体重命名参照以下规则，整体的权重转换脚本参照`/mindformers/models/llama/convert_weight.py`。

```python
def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')
    return name
```

## 5 LoRA改造

- 根据需要加入lora的线性层，构造lora匹配规则。然后在模型配置文件中`run_llama_7b_lora.yaml`将需要替换的普通线性层添加到配置项`target_modules`中。
- 定义lora rank矩阵的超参

```yaml
model:
  model_config:
    type: LlamaConfig
    ...
    pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
  arch:
    type: LlamaForCausalLM
```

## 6 训练调试

### 6.1 配置config文件

- 在config/llama/中创建yaml配置文件：（以llama 7b为例）

 ```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: ''
src_strategy_path_or_dir: ''
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'train'

# trainer config
trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'llama_7b'

# runner config
runner_config:
  epochs: 2
  batch_size: 4
  sink_mode: True
  sink_size: 2

# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.95
  eps: 1.e-8 # 1e-8
  learning_rate: 3.e-4

# lr schedule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 3.e-4
  lr_end: 3.e-5
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids"]  # "input_ids", "labels" , labels are used in instruction finetune.
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 4
  repeat: 1
  numa_enable: False
  prefetch_size: 1
train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset
# if True, do evaluate during the training process. if false, do nothing.
# note that the task trainer should support _evaluate_in_training function.
do_eval: False
eval_step_interval: -1        # num of step intervals between each eval, -1 means no step end eval.
eval_epoch_interval: 50        # num of epoch intervals between each eval, 1 means eval on every epoch end.

# eval dataset
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: False
  input_columns: ["input_ids"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: False
  repeat: 1
  numa_enable: False
  prefetch_size: 1
eval_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *eval_dataset

# default parallel of device num = 8 for Atlas 800
parallel_config:
  data_parallel: 2
  model_parallel: 1
  pipeline_stage: 4
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

# recompute config
recompute_config:
  recompute: True
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: True
  recompute_slice_activation: True

# callbacks
callbacks:
  - type: MFLossMonitor
  - type: CheckpointMonitor
    prefix: "llama_7b"
    save_checkpoint_steps: 100
    integrated_save: False
    async_save: False
  - type: ObsMonitor

# mindspore context init config
context:
  mode: 0 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  enable_graph_kernel: False
  max_call_depth: 10000
  max_device_memory: "31GB"
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0

# parallel context config
parallel:
  parallel_mode: 1 # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
  gradients_mean: False
  enable_alltoall: False
  full_batch: True
  search_mode: "sharding_propagation"
  enable_parallel_optimizer: True
  strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
  parallel_optimizer_config:
    gradient_accumulation_shard: False
    parallel_optimizer_threshold: 64

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 2048
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    vocab_size: 32000
    multiple_of: 256
    rms_norm_eps: 1.0e-6
    bos_token_id: 1
    eos_token_id: 2
    pad_token_id: 32000
    ignore_token_id: -100
    compute_dtype: "float16"
    layernorm_compute_type: "float32"
    softmax_compute_type: "float32"
    rotary_dtype: "float16"
    param_init_type: "float16"
    use_past: False
    pretrain_seqlen: 2048 # seqlen of the pretrain checkpoint: 2048 for llama and 4096 for llama2
    extend_method: "None"
    offset: 0
    checkpoint_name_or_path: "llama_7b"
    repetition_penalty: 1
    max_decode_length: 512
    top_k: 3
    top_p: 1
    do_sample: False
  arch:
    type: LlamaForCausalLM

processor:
  return_tensors: ms
  tokenizer:
    unk_token: '<unk>'
    bos_token: '<s>'
    eos_token: '</s>'
    pad_token: '<pad>'
    type: LlamaTokenizer
  type: LlamaProcessor

# metric
metric:
  type: PerplexityMetric

# wrapper cell config
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense:
    type: DynamicLossScaleUpdateCell
    loss_scale_value: 65536
    scale_factor: 2
    scale_window: 1000
  use_clip_grad: True

eval_callbacks:
  - type: ObsMonitor

auto_tune: False
filepath_prefix: './autotune'
autotune_per_step: 10

profile: False
profile_start_step: 1
profile_stop_step: 10
init_start_profile: False
profile_communication: False
profile_memory: True
layer_scale: False
layer_decay: 0.65
lr_scale_factor: 256

# aicc
remote_save_url: "Please input obs url on AICC platform."
 ```

### 6.2 启动训练和推理调试

- 通过启动run_mindformer.py进行训练、微调与推理。详细单卡、多卡使用方法参考docs/model_cards中的使用文档。

#### 单机多卡启动

1. 在仓库主目录下，运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE。

```bash
# 以八卡运行为例，生成0~7卡的hccl json文件,不包含8本身.
python ./mindformers/tools/hccl_tools.py --device_num [0,8)
```

2. 修改模型对应的配置文件。
3. 进入scripts文件夹，启动运行脚本，进行8卡分布式运行。

```bash
cd scripts
bash run_distribute.sh hccl_xxxx.json ../configs/llama/run_llama_7b.yaml [0,8) train
# 脚本启动格式：
bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_MODE]

# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的llama/run_llama_7b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8)为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

#### 多机多卡启动

1. 首先参考单机多卡启动方式，在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件。
2. 合并每台机器上生成的RANK_TABLE_FILE。
3. 将合并后的RANK_TABLE_FILE文件拷贝到所有机器中，保证不同机器上的RANK_TABLE_FILE相同。
4. 根据服务器节点数等信息，修改相应的配置。
5. 执行运行脚本。

```bash
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama/run_llama_7b.yaml [0,8) train 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/llama/run_llama_7b.yaml [8,16) train 16
```
