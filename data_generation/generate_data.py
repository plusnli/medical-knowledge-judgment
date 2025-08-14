import os
import sys
import requests
import json
import time
import openai
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Tuple
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.umls_functions import UMLS_API
from src.utils import load_json, write_json
from src.main import call_openai_api

client = openai.OpenAI(api_key="your-openai-api-key")

def from_predicates_to_templates():
    triplets_data = load_json("data_generation/triplets_collection.json")
    temp_res = {}
    
    predicates_triplets = []
    seen_pred = set()
    for dp in triplets_data:
        if dp['triplet']['predicate'] not in seen_pred:
            predicates_triplets.append(dp)
            seen_pred.add(dp['triplet']['predicate'])
    assert len(predicates_triplets) == len(seen_pred)

    for dp in tqdm(predicates_triplets):
        prompt = """Now given a triplet, your task is to return a template that can be used to generate a statement about the triplet. The statement should describe the object of the triple as the subject.
Please give your template in this format: "Template: [Your template here]".
Here are some examples.
# Example 1
Triplet: {"subject": "Peripheral Neuropathy", "predicate": "disease has associated anatomic site", "object": "Nervous System"}
In this triplet, the object is the anatomic site of the disease, and the subject is the disease. Therefore, the template should be:
Template: The anatomic site {object} has associated with the disease {subject}.

# Example 2
Triplet: {"subject": "Burning mouth syndrome", "predicate": "has finding site", "object": "Mouth region structure"}
In this triplet, the object is the finding site of the disease, and the subject is the disease. Therefore, the template should be:
Template: The {object} is in the finding site of the {subject}.

# Example 3
Triplet: {"subject": "Respiratory Depression", "predicate": "has contraindicated drug", "object": "codeine anhyd"}
In this triplet, the object is the contraindicated drug, and the subject is the disease. Therefore, the template should be:
Template: The {object} is a contraindicated drug for the {subject}.

# Now it is your turn.
"""
        prompt += f"Triplet: {dp['triplet']}"
        msg = [{'role': 'user', 'content': prompt}]
        response = call_openai_api(client=client, msg=msg, model="gpt-4o-mini", temperature=0)
        res = response.strip().split("Template:")[-1].strip()
        temp_res[dp['triplet']['predicate']] = res
    write_json("data_generation/templates.json", temp_res)


def gen_data_with_templates():
    triplets_data = load_json("data_generation/triplets_collection.json")
    semantic_to_entity = load_json("data_generation/semantic_to_entity.json")
    tempaltes = load_json("data_generation/templates.json")
    umls_api = UMLS_API("your-umls-api-key")
    random.shuffle(triplets_data)
    gen_data, obj_no_cui = [], []
    for i, dp in enumerate(tqdm(triplets_data)):
        if dp['triplet']['predicate'] not in tempaltes:
            continue
        # For positive question
        template = tempaltes[dp['triplet']['predicate']]
        pos_q = template.format(subject=dp['triplet']['subject'], object=dp['triplet']['object'])

        # For negative questions (three per predicate)
        neg_qs = []
        # get the cui of object
        cuis = umls_api.search_cui(dp['triplet']['object'])
        if len(cuis) == 0:
            print(f"Object entity not found in UMLS for dp {i}.")
            obj_no_cui.append(i)
            continue
        cui = cuis[0][0]
        name = cuis[0][1]
        semantic_types = umls_api.get_semantic_types(cui)
        type_names = [semantic_type['name'] for semantic_type in semantic_types]
        related_terms = []
        for type_name in type_names:
            related_terms.extend(semantic_to_entity[type_name])
        
        if len(related_terms) < 10:
            continue

        random.shuffle(related_terms)
        for term in related_terms[:3]:
            neg_q = template.format(subject=dp['triplet']['subject'], object=term)
            neg_qs.append(neg_q)

        gen_data.append({'original_triplet': dp['triplet']})
        gen_data[-1]['semantic_type'] = type_names
        gen_data[-1]['pos_qa'] = {'statement': pos_q, 'answer': True}
        gen_data[-1]['neg_qa'] = {'statement': neg_qs, 'answer': False}
    write_json("data_generation/binary_questions/data_with_templates.json", gen_data)


def from_entities_to_semantic_types(entity_data, entity_to_semantic_path):
    '''
    Construct mapping semantic type to entity
    1. get entity's cui; 2. get semantic type; 3. establish mapping
    '''
    semantic_to_entity = {}
    no_cui_cnt, no_semantic_cnt = 0, 0
    names = []
    umls_api = UMLS_API("your-umls-api-key")
    for i, entity in enumerate(tqdm(entity_data)):
        cuis = umls_api.search_cui(entity)  # get the cui of object
        if len(cuis) == 0:
            print(f"Not found CUI for {entity}")
            no_cui_cnt += 1
            continue
        cui = cuis[0][0]
        name = cuis[0][1]
        names.append(name)
        # assert name == entity
        names = list(set(names))
        print(f"Len of unique names: {len(names)}")

        semantic_types = umls_api.get_semantic_types(cui)
        if not semantic_types:
            print(f"Not found semantic types for {entity}")
            no_semantic_cnt += 1
            continue
        for semantic_type in semantic_types:
            if semantic_type['name'] not in semantic_to_entity:
                semantic_to_entity[semantic_type['name']] = [name]
            elif semantic_type['name'] in semantic_to_entity:
                semantic_to_entity[semantic_type['name']].append(name)
    write_json(entity_to_semantic_path, semantic_to_entity)


