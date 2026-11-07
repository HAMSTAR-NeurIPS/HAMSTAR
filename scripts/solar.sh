#!/bin/bash
# Solar -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_solar_96 \
  --data custom \
  --root_path ./dataset/ \
  --data_path solar.csv \
  --target 136 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 137 \
  --period_len 144 \
  --emb_num 80 \
  --e_layers 2 \
  --n_steps 7 \
  --dropout 0.01 \
  --learning_rate 0.005 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_solar_192 \
  --data custom \
  --root_path ./dataset/ \
  --data_path solar.csv \
  --target 136 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 137 \
  --period_len 144 \
  --emb_num 112 \
  --e_layers 1 \
  --n_steps 2 \
  --dropout 0.01 \
  --learning_rate 0.005 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_solar_336 \
  --data custom \
  --root_path ./dataset/ \
  --data_path solar.csv \
  --target 136 \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 137 \
  --period_len 144 \
  --emb_num 224 \
  --e_layers 2 \
  --n_steps 3 \
  --dropout 0.2 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_solar_720 \
  --data custom \
  --root_path ./dataset/ \
  --data_path solar.csv \
  --target 136 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 137 \
  --period_len 144 \
  --emb_num 208 \
  --e_layers 3 \
  --n_steps 4 \
  --dropout 0.15 \
  --learning_rate 0.005 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
