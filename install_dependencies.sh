#!/bin/bash
set -e

conda env create --file ./environment.yml
conda run -n gs_init_compare pip install -r requirements.txt
