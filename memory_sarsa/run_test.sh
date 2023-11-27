#!/bin/bash

for i in $(seq 1 2)
do
    python test.py --seed $i --row 3 --col 3
done