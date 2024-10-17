#!/bin/bash

num=40

for ((i=1; i<=num; i++))
do
    python main.py --radius 5 --file-name nr_small_$i --num-episode-steps 100000 --lr 8e-4 --mode train
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 18 --file-name nr_small_large_$i --num-episode-steps 20000 --lr 4e-4 --model-name nr_small_$i --mode transfer
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 5 --file-name nr_small_small_$i --num-episode-steps 20000 --lr 4e-4 --model-name nr_small_$i --mode transfer
done