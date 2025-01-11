#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <scene> <model>"
    exit 1
fi

scene=$1
model=$2

nerfbaselines viewer --checkpoint nerfbaselines_results/$scene/$model/checkpoint-30000/ --data external://$scene --backend python --port 0
