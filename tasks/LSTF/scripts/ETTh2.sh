### M
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

### S
python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5
