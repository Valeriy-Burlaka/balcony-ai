#!/usr/bin/env zsh

ENV_DIR=.tf-venv

if [[ ! -d $ENV_DIR ]]; then
    python -m venv $ENV_DIR
fi

source $ENV_DIR/bin/activate

pip install --upgrade pip

# pip install -r requirements.txt
pip install numpy==1.26.4
