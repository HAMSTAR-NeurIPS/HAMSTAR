#!/bin/bash
# ETTm2 -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_ettm2_96 \
  --data ETTm2 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --period_len 96 \
  --emb_num 224 \
  --e_layers 2 \
  --n_steps 6 \
  --dropout 0.05 \
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
  --model_id HAMSTAR_ettm2_192 \
  --data ETTm2 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --period_len 4 \
  --emb_num 240 \
  --e_layers 2 \
  --n_steps 3 \
  --dropout 0.1 \
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
  --model_id HAMSTAR_ettm2_336 \
  --data ETTm2 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --period_len 4 \
  --emb_num 128 \
  --e_layers 1 \
  --n_steps 5 \
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
  --model_id HAMSTAR_ettm2_720 \
  --data ETTm2 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --period_len 4 \
  --emb_num 16 \
  --e_layers 3 \
  --n_steps 7 \
  --dropout 0.1 \
  --learning_rate 0.005 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
