#!/bin/bash
# PM2.5 -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_pm2_5_96 \
  --data custom \
  --root_path ./dataset/ \
  --data_path pm2_5.csv \
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
  --model_id HAMSTAR_pm2_5_192 \
  --data custom \
  --root_path ./dataset/ \
  --data_path pm2_5.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
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
  --model_id HAMSTAR_pm2_5_336 \
  --data custom \
  --root_path ./dataset/ \
  --data_path pm2_5.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 32 \
  --e_layers 2 \
  --n_steps 7 \
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
  --model_id HAMSTAR_pm2_5_720 \
  --data custom \
  --root_path ./dataset/ \
  --data_path pm2_5.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 184 \
  --period_len 8 \
  --emb_num 32 \
  --e_layers 2 \
  --n_steps 7 \
  --dropout 0.01 \
  --learning_rate 0.005 \
  --batch_size 16 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
