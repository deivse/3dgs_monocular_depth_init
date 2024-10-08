#!/bin/sh

git submodule init
GIT_LFS_SKIP_SMUDGE=1 git submodule update

poetry config virtualenvs.in-project true
poetry install
