### M
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

### S
python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5
