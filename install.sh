#!/bin/bash
set -e

git submodule update --init --recursive

# Take env name from first argument or use default
ENV_NAME="${1:-gs_init_compare}"

echo "Using environment name '$ENV_NAME'"

if conda env list | grep -q $ENV_NAME; then
    echo "Environment '$ENV_NAME' already exists, updating..."
    conda env update --file ./environment.yml --prune --name $ENV_NAME
else
    echo "Creating environment '$ENV_NAME'..."
    conda env create --file ./environment.yml --name $ENV_NAME
fi
conda run --live-stream -n $ENV_NAME pip install --editable .
