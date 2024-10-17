#!/bin/bash

num=2

for ((i=1; i<=num; i++))
do
    python main.py --radius 12 --file-name large_$i --num-episode-steps 100000 --lr 8e-4 --mode train --exp-name sigmoid
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 12 --file-name large_large_$i --num-episode-steps 20000 --lr 4e-4 --model-name large_$i --mode transfer --exp-name sigmoid
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 6 --file-name large_small_$i --num-episode-steps 20000 --lr 4e-4 --model-name large_$i --mode transfer --exp-name sigmoid
done