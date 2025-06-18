#!/bin/bash

# # make sure that there is a .venv 
# if [ ! -d ".venv" ]; then
# 	echo ".venv does not exsist. Generating one now..."
# 	python3 -m venv .venv
# fi
# 
# # use the venv
# source .venv/bin/activate
# 
# # install required packages
# if [ -f "requirements.txt" ]; then
#     echo "Installing dependencies from requirements.txt..."
#     pip install --upgrade pip
#     pip install -r requirements.txt
# else
#     echo "[WARNING] requirements.txt not found! Skipping dependency installation."
# fi
python ./src/testing_main.py 
