#!/bin/bash
# Electricity -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_electricity_96 \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 321 \
  --period_len 24 \
  --emb_num 80 \
  --e_layers 2 \
  --n_steps 7 \
  --dropout 0.01 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_electricity_192 \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 321 \
  --period_len 24 \
  --emb_num 160 \
  --e_layers 2 \
  --n_steps 7 \
  --dropout 0.01 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_electricity_336 \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 321 \
  --period_len 24 \
  --emb_num 176 \
  --e_layers 2 \
  --n_steps 5 \
  --dropout 0.01 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_electricity_720 \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 321 \
  --period_len 24 \
  --emb_num 176 \
  --e_layers 2 \
  --n_steps 5 \
  --dropout 0.01 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
