#!/bin/bash

model=$1

# For vanilla evaluation (zero-shot)
python src/main.py --model $model

# For retrieval evaluation
python src/main.py --model $model --retrieval sparse
python src/main.py --model $model --retrieval dense

# For uncertainty evaluation
python src/main.py --model $model --uncertainty
