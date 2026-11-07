#!/bin/bash
# Weather -- HAMSTAR main-table reproduction
set -e

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_weather_96 \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 21 \
  --period_len 144 \
  --emb_num 240 \
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

python -u run_longExp.py \
  --is_training 1 \
  --model HAMSTAR \
  --model_id HAMSTAR_weather_192 \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 21 \
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
  --model_id HAMSTAR_weather_336 \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 21 \
  --period_len 144 \
  --emb_num 256 \
  --e_layers 2 \
  --n_steps 6 \
  --dropout 0.01 \
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
  --model_id HAMSTAR_weather_720 \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --target OT \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 21 \
  --period_len 144 \
  --emb_num 16 \
  --e_layers 2 \
  --n_steps 6 \
  --dropout 0.1 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10 \
  --use_orthogonal 0 \
  --lradj type3 \
  --itr 1
