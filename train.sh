#!/bin/bash

# 数据处理
python3 dataProcess.py

# 1、训练
torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir output/bge-large-en-v1.5-ft \
--model_name_or_path pretrain_models/bge-large-en-v1.5 \
--train_data data/bge_ft/bge_ft.json \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 10 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 256 \
--passage_max_len 256 \
--train_group_size 8 \
--negatives_cross_device \
--logging_steps 500 \
--report_to tensorboard \
--query_instruction_for_retrieval "" \
--save_total_limit 5