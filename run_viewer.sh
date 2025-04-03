#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <scene> <model> [<port=0>]"
    exit 1
fi

scene=$1
model=$2
port=${3:-0}

RESULTS_DIR="${RESULTS_DIR:-nerfbaselines_results}"

nerfbaselines viewer --checkpoint $RESULTS_DIR/$scene/$model/checkpoint-30000/ --data external://$scene --backend python --port $port
