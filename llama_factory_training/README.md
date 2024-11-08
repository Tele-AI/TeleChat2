
# llama factory 适配
## 适配流程

### 环境安装
参考llama factory官方文档
https://github.com/hiyouga/LLaMA-Factory/tree/main 

### 模型下载
首先下载需要微调的telechat模型，例如模型所在位置为：
./telechat2-7B
其中包含以下文件：
1. config.json                 
2. modeling_telechat2.py               
3. tokenizer.model
4. configuration_telechat2.py            
5. tokenizer_config.json
6. generation_config.json      
7. pytorch_model.bin.index.json 
8. tokenization_telechat2.py     
9. generation_utils.py        
10. pytorch_model_00001-of-00004.bin  
11. pytorch_model_00002-of-00004.bin
12. pytorch_model_00003-of-00004.bin
13. pytorch_model_00004-of-00004.bin

### 修改模型文件
需要将模型位置中的"modeling_telechat2.py"文件进行替换，使用文档中[该文件](./modeling_telechat2.py)替换即可。


### 微调代码运行
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \--stage sft \--do_train True \--model_name_or_path ./telechat2-7B \--preprocessing_num_workers 16 \--finetuning_type lora \--template telechat \--flash_attn auto \--dataset_dir data \--dataset identity,alpaca_en_demo \--cutoff_len 1024 \--learning_rate 5e-05 \--num_train_epochs 3.0 \--max_samples 100000 \--per_device_train_batch_size 2 \--gradient_accumulation_steps 8 \--lr_scheduler_type cosine \--max_grad_norm 1.0 \--logging_steps 5 \--save_steps 100 \--warmup_steps 0 \--optim adamw_torch \--packing False \--report_to none \--output_dir saves/TeleChat-1B-Chat/lora/train_2024-08-01-10-20-02 \--plot_loss True \--ddp_timeout 180000000 \--include_num_input_tokens_seen True \--lora_rank 8 \--lora_alpha 16 \--lora_dropout 0 \--lora_target all  

注：不能支持bf16精度进行微调，因此不能出现该参数：--bf16 True


