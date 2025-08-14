<div align="center">

<h1>🩺 Medical Knowledge Judgment (MKJ)</h1>

<h3>
  <a href="https://arxiv.org/abs/2502.14275">
    Fact or Guesswork? <br>
    Evaluating Large Language Model's Medical Knowledge <br>
    with Structured One-Hop Judgment
  </a>
</h3>

</div>


## 📂 Repository Structure

### 💻 Core Source Code (`src/`)

- **`main.py`**: Main file for evaluation on the MKJ dataset:
  - Vanilla prompting (zero-shot), RAG (Retrieval-Augmented Generation) with sparse/dense retrieval, and Uncertainty calibration analysis
- **`umls_functions.py`**: UMLS API wrapper providing medical knowledge retrieval capabilities.
- **`utils.py`**: File I/O operations.
- **`pubmed_utils.py`**: PubMed database API integration for term frequency analysis.

### 🏭 Data Generation (`data_generation/`)

#### Scripts:
- **`collect_triplets.py`**: UMLS triplet collection and processing.
- **`generate_data.py`**: Primary data generation pipeline based on collected triplets.

#### Data Files:
- **`triplets_collection.json`**: Collection of UMLS triplets.
- **`templates.json`**: Templates for datapoint generation.
- **`semantic_to_entity.json`**: Mapping between semantic types and medical entities in UMLS.


### 📝 Data Directory (`data/`)

- **`data.json`**: Sample dataset for medical knowledge evaluation.
- **`hf_dataset/`**: In huggingface dataset format.

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### API Keys Required

The evaluation requires three API keys:
- OpenAI API key
- Anthropic Claude API key
- UMLS Terminology Services API key

### Basic Usage

```bash
# Vanilla evaluation
python src/main.py --model "model-name"

# RAG evaluation
python src/main.py --model "model-name" --retrieval sparse

# Uncertainty analysis
python src/main.py --model "model-name" --uncertainty

# Batch evaluation
bash run.sh "model-name"
```

Our framework supports a wide range of models (OpenAI, Anthropic Claude, general-domain LLMs, and medical LLMs) as follows (see `src/main.py` for details):

**OpenAI Models:**
- gpt-3.5-turbo
- gpt-4o-mini
- gpt-4o

**Anthropic Claude Models:**
- claude-3-haiku-20240307
- claude-3-sonnet-20240229
- claude-3-5-sonnet-20240620

**Open-source Models (from Huggingface):**
- llama-3-8B-instruct
- llama-3.1-8B-instruct
- llama-3.2-1B-instruct
- llama-3.2-3B-instruct
- mistral-8B-instruct
- qwen2.5-0.5B-instruct
- qwen2.5-1.5B-instruct
- qwen2.5-3B-instruct
- phi-3-mini-4k-instruct
- phi-3-mini-128k-instruct
- phi-3.5-mini-instruct
- llama-2-7B-chat
- llama-2-13B-chat

**Medical LLMs:**
- meditron-7B (from Huggingface)
- mellama-13B, which needs to be downloaded from the [website](https://www.physionet.org/content/me-llama/1.0.0/) and placed in the `me-llama/1.0.0/` directory

For more details or to add support for additional models, please refer to the model configuration and argument parsing in `src/main.py`.

## 📊 Dataset

Due to UMLS privacy policy restrictions, a sample dataset is provided in `data/data.json` for testing and demonstration purposes. We also have the corresponding huggingface format in `data/hf_dataset/`.

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{li2025fact,
  title={Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment},
  author={Li, Jiaxi and Wang, Yiwei and Zhang, Kai and Cai, Yujun and Hooi, Bryan and Peng, Nanyun and Chang, Kai-Wei and Lu, Jin},
  journal={arXiv preprint arXiv:2502.14275},
  year={2025}
}
```

