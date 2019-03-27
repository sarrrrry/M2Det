#!/usr/bin/env bash
cd ./utils/

pipenv run python build.py build_ext --inplace

cd ..
