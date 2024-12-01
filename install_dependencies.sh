#!/bin/bash
set -e

git submodule update --init --recursive

if conda env list | grep -q 'gs_init_compare'; then
    echo "Environment 'gs_init_compare' already exists, updating..."
    conda env update --file ./environment.yml --prune
else
    echo "Creating environment 'gs_init_compare'..."
    conda env create --file ./environment.yml
fi
conda run --live-stream -n gs_init_compare pip install -r requirements.txt
