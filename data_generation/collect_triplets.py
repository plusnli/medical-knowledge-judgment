import os
import torch
import cohere
import numpy as np
import re
import json
import requests
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

device = torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts: List, batch_size: int=16):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                model_output = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self.mean_pooling(model_output, attention_mask)
            all_embeddings.extend(batch_embeddings.cpu().numpy())
        return np.array(all_embeddings)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

class UMLS_CohereReranker:
    def __init__(self, api_key):
        self.co = cohere.Client(api_key)
    
    def rerank(self, query, rels, triplets, top_n=20):
        results = self.co.rerank(model="rerank-english-v3.0", query=query, documents=rels, top_n=top_n)
        results_list = results.results
        reranked_results = []
        for r in results_list:
            if r.document is None:
                reranked_results.append({
                    "rel": rels[r.index],
                    "triplet": triplets[r.index],
                    "relevance_score": r.relevance_score
                })
            else:
                reranked_results.append({
                    "rel": r.document["text"],
                    "triplet": triplets[r.index],
                    "relevance_score": r.relevance_score
                })
        return reranked_results

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_concepts(self, cui):
        try:
            concept_suffix = "/CUI/{}?apiKey={}"
            suffix = concept_suffix.format(cui, self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_semantic_types(self, cui):
        res = self.get_concepts(cui)
        if res is None:
            return None
        semantic_types = res['semanticTypes']
        return semantic_types

    def get_definitions(self, cui):
        print("*****UMLS Getting Definitions*****")
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui, pages=20):
        print("*****UMLS Getting Relations*****")
        all_relations = []

        try:
            for page in tqdm(range(1, pages + 1), desc="Getting relations in pages..."):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations


def get_umls_keys(query, prompt, llm, umls_api, umlsbert, cohere_reranker):
    umls_res = {}
    prompt = prompt.replace("{question}", query)
    keys_text = llm.invoke(prompt).content
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, keys_text.replace("\n", ""))

    if not matches:
        raise ValueError("No medical terminologies returned by the model.")
    try: 
        keys_dict = json.loads("{" + matches[0] + "}")
    except Exception as e:
        print(f"Error during loading the data pattern into json format.\n{e}")
        return ""
    
    if "medical_terminologies" not in keys_dict or not keys_dict["medical_terminologies"]:
        raise ValueError("Model did not return expected 'medical_terminologies' key.")

    for key in keys_dict["medical_terminologies"][:]:
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            continue
        cui = cuis[0][0]
        name = cuis[0][1]

        defi = ""
        definitions = umls_api.get_definitions(cui)

        if definitions is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition["rootSource"]
                if source == "MSH":
                    msh_def = definition["value"]
                    break
                elif source == "NCI":
                    nci_def = definition["value"]
                elif source == "ICF":
                    icf_def = definition["value"]
                elif source == "CSP":
                    csp_def = definition["value"]
                elif source == "HPO":
                    hpo_def = definition["value"]

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        relations = umls_api.get_relations(cui)
        rels = []
        if relations is not None:
            relation_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]

            embeddings = umlsbert.batch_encode(relation_texts)
            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]
            relation_scores = [(get_similarity([query_embedding], [rel_emb]), rel) for rel_emb, rel in zip(relation_embeddings, relations)]
            relation_scores.sort(key=lambda x: x[0], reverse=True)
            top_rels = relation_scores[:200]

            top_rel_texts = [f"{rel[1].get('relatedFromIdName', '')} {rel[1].get('additionalRelationLabel', '').replace('_', ' ')} {rel[1].get('relatedIdName', '')}" for rel in top_rels]
            triplets = [{'subject': rel[1].get('relatedFromIdName', ''), 'predicate': rel[1].get('additionalRelationLabel', '').replace('_', ' '), 'object': rel[1].get('relatedIdName', '')} for rel in top_rels]
            if top_rel_texts:
                reranked_rel_results = cohere_reranker.rerank(query, top_rel_texts, triplets)
            else:
                reranked_rel_results = []
            for result in reranked_rel_results:
                rels.append(result)

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    context_dict = {}
    context_txt = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        context_dict[k] = v

        rels_text = ""
        for rel in rels:
            relation_description = rel['rel']
            rels_text += f"({relation_description})\n"
        text = f"Name: {name}\nDefinition: {definition}\n"
        if rels_text != "":
            text += f"Relations: \n{rels_text}"
        context_txt += text + "\n"
    if context_txt != "":
        context_txt = context_txt[:-1]

    return context_dict, context_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to process")
    parser.add_argument("--openai_api", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--umls_api", type=str, required=True, help="UMLS API key")
    parser.add_argument("--cohere_api", type=str, required=True, help="Cohere API key")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory for results")
    args = parser.parse_args()

    # Initialize components
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=args.openai_api)
    umls_api = UMLS_API(args.umls_api)
    umlsbert = UMLSBERT()
    cohere_reranker = UMLS_CohereReranker(args.cohere_api)

    KEYWORD_PROMPT = """
Question: {question}

You are interacting with a knowledge graph that contains definitions and relational information of medical terminologies. To provide a precise and relevant answer to this question, you are expected to:

1. Understand the Question Thoroughly: Analyze the question deeply to identify which specific medical terminologies and their interrelations, as extracted from the knowledge graph, are crucial for formulating an accurate response.

2. Extract Key Terminologies: Return the 3-5 most relevant medical terminologies based on their significance to the question.

3. Format the Output: Return in a structured JSON format with the key as "medical_terminologies". For example:
[
    {
        "medical_terminologies": ["term1", "term2", ...]
    }
]
"""

    # Load dataset
    ds = load_dataset(args.dataset, split='test')
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_file = f"{args.output_dir}/{args.dataset.replace('/', '_')}_umls_terms.json"
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        results = load_json(output_file)
        print(f"Loaded {len(results)} existing results")
    else:
        results = []

    print(f"Processing {len(ds['test'])} questions from dataset: {args.dataset}")
    
    for i in tqdm(range(len(ds['test'])), desc="Processing Questions"):
        # Skip if already processed
        if i < len(results):
            continue
            
        q = ds['test'][i]['NLM_SUMMARY']
        
        try:
            context_dict, context_txt = get_umls_keys(q, KEYWORD_PROMPT, llm, umls_api, umlsbert, cohere_reranker)
            results.append({
                "question_id": i,
                "question": q, 
                "umls_context_dict": context_dict, 
                "umls_context_txt": context_txt
            })
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({
                "question_id": i,
                "question": q, 
                "umls_context_dict": {}, 
                "umls_context_txt": "",
                "error": str(e)
            })
        
        # Save results every 5 iterations
        if (i + 1) % 5 == 0:
            write_json(output_file, results)
            print(f"Saved results up to question {i}")
    
    # Final save
    write_json(output_file, results)
    print(f"Completed processing. Results saved to: {output_file}")
    