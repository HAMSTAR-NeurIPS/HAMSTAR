#!/bin/bash
# ETTh2 -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_etth2_96 \
  --data ETTh2 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --period_len 24 \
  --emb_num 80 \
  --e_layers 3 \
  --n_steps 2 \
  --dropout 0.15 \
  --learning_rate 0.002 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_etth2_192 \
  --data ETTh2 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --period_len 24 \
  --emb_num 64 \
  --e_layers 1 \
  --n_steps 8 \
  --dropout 0.2 \
  --learning_rate 0.002 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_etth2_336 \
  --data ETTh2 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --period_len 24 \
  --emb_num 112 \
  --e_layers 2 \
  --n_steps 6 \
  --dropout 0.01 \
  --learning_rate 0.001 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_etth2_720 \
  --data ETTh2 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --period_len 24 \
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
