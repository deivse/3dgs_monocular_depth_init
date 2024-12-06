cd src
conda run --live-stream -n gs_init_compare python evaluator.py \
    --downsample-factors 10 \
    --max-steps 15000
