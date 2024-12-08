#!/bin/sh

echo ========================================
echo Installing pip dependencies...
echo ========================================

conda run --live-stream -n gs_init_compare pip install --editable .

echo == Welcome... ==
