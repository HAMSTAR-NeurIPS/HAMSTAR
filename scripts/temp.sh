#!/bin/bash
# Temp -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_temp_96 \
  --data custom \
  --root_path ./dataset/ \
  --data_path temp.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 112 \
  --e_layers 1 \
  --n_steps 2 \
  --dropout 0.01 \
  --learning_rate 0.005 \
  --batch_size 16 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_temp_192 \
  --data custom \
  --root_path ./dataset/ \
  --data_path temp.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 96 \
  --e_layers 3 \
  --n_steps 4 \
  --dropout 0.1 \
  --learning_rate 0.005 \
  --batch_size 16 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_temp_336 \
  --data custom \
  --root_path ./dataset/ \
  --data_path temp.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 240 \
  --e_layers 3 \
  --n_steps 5 \
  --dropout 0.2 \
  --learning_rate 0.01 \
  --batch_size 16 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_temp_720 \
  --data custom \
  --root_path ./dataset/ \
  --data_path temp.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 160 \
  --e_layers 3 \
  --n_steps 4 \
  --dropout 0.05 \
  --learning_rate 0.01 \
  --batch_size 16 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
