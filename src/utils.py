import os
import sys
import random
import re
import json
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def read_txt(path):
    with open(path, "r", encoding='utf-8') as f:
        data = f.read()
    return data

def write_txt(path, data):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, "w+") as f:
        f.write(data)
