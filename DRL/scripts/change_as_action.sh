#!/bin/bash

num=1

for ((i=1; i<=num; i++))
do
    python main.py --radius 6 --file-name small_$i --num-episode-steps 100000 --lr 8e-4 --mode train --exp-name change_as_action
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 12 --file-name small_large_$i --num-episode-steps 20000 --lr 4e-4 --model-name small_$i --mode transfer --exp-name change_as_action
done

for ((i=1; i<=num; i++))
do
    python main.py --radius 6 --file-name small_small_$i --num-episode-steps 20000 --lr 4e-4 --model-name small_$i --mode transfer --exp-name change_as_action
done