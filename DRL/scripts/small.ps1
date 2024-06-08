$num = 50

for ($i = 1; $i -le $num; $i++) {
    python main.py --radius 5 --file-name small_$i --num-episode-steps 100000 --lr 8e-4 --mode train
}

for ($i = 1; $i -le $num; $i++) {
    python main.py --radius 18 --file-name small_large_$i --num-episode-steps 20000 --lr 4e-4 --model-name small_$i --mode transfer
}

for ($i = 1; $i -le $num; $i++) {
    python main.py --radius 5 --file-name small_small_$i --num-episode-steps 20000 --lr 4e-4 --model-name small_$i --mode transfer
}