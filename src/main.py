import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import openai
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.calibration import calibration_curve
from typing import Tuple, List
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import argparse

import umls_functions
from utils import load_json, write_json


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
def call_openai_api(client, msg, model, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=temperature
    )
    output = response.choices[0].message.content
    return output

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
def call_claude_api(client, msg, model, temperature):
    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=temperature,
        messages=msg
    )
    res = message.content[0].text
    return res

def run_open_source_models(pipeline, msg):
    outputs = pipeline(
        msg,
        max_new_tokens=1024,
    )
    res = outputs[0]["generated_text"][-1]["content"]
    return res


def run_models(args, prompt):
    msg = [{'role': 'system', 'content': 'You are a helpful chatbot.'}, {'role': 'user', 'content': prompt}]
    if args.model_type == "open-source":
        response = run_open_source_models(pipeline=pipeline, msg=msg)
    elif 'gpt' in args.model:
        response = call_openai_api(client=client, msg=msg, model=args.model, temperature=args.temperature)
    elif 'claude' in args.model:
        response = call_claude_api(client=client, msg=msg, model=args.model, temperature=args.temperature)
    return response


def eval_dp(args, query):
    """evaluate model's performance on a single query with zero-shot prompting."""
    prompt = "Given a statement, your task is to judge it true or false. "
    prompt += f"""Finally, format your answer like this: "Answer: [True or False]".\nStatement: {query}"""
    response = run_models(args, prompt)
    pred = None
    response_lower = response.lower()
    true_idx = response_lower.rfind('true')
    false_idx = response_lower.rfind('false')
    if true_idx == -1 and false_idx == -1:
        pred = False
    elif true_idx > false_idx:
        pred = True
    else:
        pred = False
    return response, pred

def rag(args, query):
    triplets_data = load_json("data_generation/triplets_collection.json")
    docs = [dp['text'] for dp in triplets_data]
    # Sparse retrieval
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    # Dense retrieval
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = encoder.encode(docs)

    def sparse_retrieval(query, top_k=5):
        query_vec = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        top_docs = [docs[i] for i in top_indices]
        return top_docs

    def dense_retrieval(query, top_k=5):
        query_embedding = encoder.encode([query])
        cosine_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        top_docs = [docs[i] for i in top_indices]
        return top_docs

    def generate_answer(query, retrieved_docs):
        context = "Documents:\n"
        context += '\n'.join(retrieved_docs)
        prompt = f"""Please judge the following statement is true or false based on the documents. Some relevant documents are provided that may help you do the judgement.
Finally, please format your answer like this: "Answer: [True or False]".\n
Statement:\n{query}\n\n{context}"""
        response = run_models(args, prompt)
        return response

    if args.retrieval == 'sparse':
        retrieved_docs = sparse_retrieval(query)
        res = generate_answer(query, retrieved_docs)
    elif args.retrieval == 'dense':
        retrieved_docs = dense_retrieval(query)
        res = generate_answer(query, retrieved_docs)
    pred = None
    res_lower = res.lower()
    true_idx = res_lower.rfind('true')
    false_idx = res_lower.rfind('false')
    if true_idx == -1 and false_idx == -1:
        pred = False
    elif true_idx > false_idx:
        pred = True
    else:
        pred = False
    return retrieved_docs, res, pred


def evaluate(args, data):
    print('*' * 10, f"Evaluating model {args.model}", '*' * 10)
    acc, pos_acc, neg_acc = 0, 0, 0
    weight = 0.25
    results = {'Accuracy': None, 'data': data}
    
    for i, dp in enumerate(tqdm(data)):
        # For pos_qa
        ans = True
        if args.retrieval:
            docs, response, pred = rag(args, dp['pos_qa']['statement'])
        else:
            response, pred = eval_dp(args, dp['pos_qa']['statement'])
        acc += weight * (pred == ans)
        pos_acc += (pred == ans)
        results['data'][i]['pos_qa']['result'] = {'response': response, 'pred': pred, 'acc': int(pred == ans)}
        if args.retrieval:
            results['data'][i]['pos_qa']['result']['docs'] = docs

        # For neg_qa
        ans = False
        neg_results = []        
        for neg_q in dp['neg_qa']['statement']:
            if args.retrieval:
                docs, response, pred = rag(args, neg_q)
            else:
                response, pred = eval_dp(args, neg_q)
            acc += weight * (pred == ans)
            neg_acc += (pred == ans)
            neg_results.append({'response': response, 'pred': pred, 'acc': int(pred == ans)})
            if args.retrieval:
                neg_results[-1]['docs'] = docs
        results['data'][i]['neg_qa']['result'] = neg_results
        if i % 10 == 0:
            write_json(args.result_path, results)
    
    acc = 1.0 * acc / (len(data))
    pos_acc = 1.0 * pos_acc / (len(data))
    neg_acc = 1.0 / 3.0 * neg_acc / (len(data)) 
    results['Accuracy'] = {'acc': acc, 'pos_acc': pos_acc, 'neg_acc': neg_acc}
    return results


def get_uncertainty(args, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = model2id[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, device_map='auto', torch_dtype=torch.bfloat16)
    model = model.to(device)
    # pad_token -> eos_token (end of sequence) if doesn't have pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    texts, y_labels = [], []
    for i, dp in enumerate(data):
        texts.append(dp['pos_qa']['statement'])
        y_labels.append(1)
        for neg_q in dp['neg_qa']['statement']:
            texts.append(neg_q)
            y_labels.append(0)
    assert len(texts) == len(y_labels)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    batch_size = 16
    cur = 0  # current index in the data list
    probabilities = []
    model.eval()
    inputs = inputs.to(device)
    while cur < len(inputs['input_ids']):
        batch_inputs = {k: v[cur: min(cur + batch_size, len(inputs['input_ids']))] for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs)
        logits = outputs.logits.to(torch.float32)
        batch_prob = torch.softmax(logits, dim=-1).cpu().numpy()  # move data from GPU to CPU (tensor to ndarray)
        probabilities.extend(batch_prob.tolist())
        cur += batch_size
    y_prob = []
    for i in range(len(probabilities)):
        y_prob.append(probabilities[i][y_labels[i]])
    y_labels = np.array(y_labels)
    y_prob = np.array(y_prob)
    n_bins = 100
    prob_true_uniform, prob_pred_uniform = calibration_curve(y_labels, y_prob, n_bins=n_bins, strategy='uniform')  # with strategy of uniform instead of quantile for better visualization

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='black')  # Diagonal line
    plt.plot(prob_pred_uniform, prob_true_uniform, marker='o', label='Model Calibration', color='blue')

    plt.xlabel('Confidence', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title(f'Calibration curve ({args.model})', fontsize=18)
    plt.legend(loc='upper left', fontsize=14)
    plt.savefig(f"results/uncertainty/calibration_curve_{args.model}_bin{n_bins}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', type=str, 
                        choices=['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-5-sonnet-20240620', 'llama-3-8B', 'llama-3.1-8B', 'llama-3.2-1B', 'llama-3.2-3B', 'ministral-8B', 'qwen2.5-0.5B', 'qwen2.5-1.5B', 'qwen2.5-3B', 'phi-3-mini-4k', 'phi-3-mini-128k', 'phi-3.5-mini', 'meditron-7B', 'mellama-13B', 'llama-2-7B', 'llama-2-13B'])
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--openai_api', type=str, default='your-openai-api-key')
    parser.add_argument("--claude_api", type=str, default='your-claude-api-key')
    parser.add_argument("--umls_api", type=str, default='your-umls-api-key')
    
    parser.add_argument("--retrieval", type=str, choices=['sparse', 'dense'])
    parser.add_argument("--uncertainty", action='store_true', help="Evaluate uncertainty.")
    args = parser.parse_args()

    open_models = ['llama', 'vicuna', 'falcon', 'mistral', 'ministral', 'qwen', 'cpm', 'phi', 'meditron', 'mellama']
    model2id = {
        'llama-3-8B': 'meta-llama/Meta-Llama-3-8B-Instruct', 
        'llama-3.1-8B': "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        'llama-3.2-1B': "meta-llama/Llama-3.2-1B-Instruct", 
        'llama-3.2-3B': "meta-llama/Llama-3.2-3B-Instruct",
        'ministral-8B': 'mistralai/Ministral-8B-Instruct-2410', 
        'qwen2.5-0.5B': 'Qwen/Qwen2.5-0.5B-Instruct', 
        'qwen2.5-1.5B': 'Qwen/Qwen2.5-1.5B-Instruct', 
        'qwen2.5-3B': 'Qwen/Qwen2.5-3B-Instruct',
        'phi-3-mini-4k': 'microsoft/Phi-3-mini-4k-instruct', 
        'phi-3-mini-128k': 'microsoft/Phi-3-mini-128k-instruct', 
        'phi-3.5-mini': 'microsoft/Phi-3.5-mini-instruct',
        'meditron-7B': 'epfl-llm/meditron-7b', 
        'mellama-13B': 'me-llama/1.0.0/MeLLaMA-13B-chat',  # Need to be downloaded from https://www.physionet.org/content/me-llama/1.0.0/.
        'llama-2-7B': 'meta-llama/Llama-2-7b-chat-hf', 
        'llama-2-13B': 'meta-llama/Llama-2-13b-chat-hf'
    }
    
    if args.uncertainty:
        data = load_json("data_generation/binary_questions/data_with_templates.json")
        get_uncertainty(args, data)
        exit(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    umls_api = umls_functions.UMLS_API(args.umls_api)
    if any(open_model in args.model for open_model in open_models):
        args.model_type = "open-source"
        model_id = model2id[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            trust_remote_code=True
        )
    else:
        args.model_type = "closed-source"
        if 'gpt' in args.model:
            client = openai.OpenAI(api_key=args.openai_api)
        elif 'claude' in args.model:
            client = anthropic.Anthropic(api_key=args.claude_api)

    data = load_json("data_generation/binary_questions/data_with_templates.json")
    # dataset = datasets.load_from_disk("data_generation/binary_questions/hf_dataset")
    if args.retrieval:
        args.result_path = f"results/rag/results_{args.model}_{args.retrieval}.json"
    else:
        args.result_path = f"results/vanilla/results_{args.model}.json"
    results = evaluate(args, data)
    print(f"Acc: {results['Accuracy']['acc']}\nPos acc: {results['Accuracy']['pos_acc']}\nNeg acc: {results['Accuracy']['neg_acc']}")
    write_json(args.result_path, results)
