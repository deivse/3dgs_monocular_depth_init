#!/bin/sh

echo ========================================
echo Installing pip dependencies...
echo ========================================

conda run --no-capture-output -n gs_init_compare pip install -r ./requirements.txt

echo == Welcome... ==
