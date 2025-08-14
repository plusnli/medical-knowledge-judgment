# Medical Knowledge Judgment (MKJ)

**Paper**: "[Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment](https://arxiv.org/abs/2502.14275)"

<!-- This paper introduces the Medical Knowledge Judgment (MKJ) dataset for assessing large language models' medical knowledge using triplets collected from UMLS (Unified Medical Language System). -->

## Repository Structure

### Core Source Code (`src/`)

- **`main.py`**: Main file for evaluation on the MKJ dataset:
  - Vanilla prompting (zero-shot), RAG (Retrieval-Augmented Generation) with sparse/dense retrieval, and Uncertainty calibration analysis
- **`umls_functions.py`**: UMLS API wrapper providing medical knowledge retrieval capabilities.
- **`utils.py`**: File I/O operations.
- **`pubmed_utils.py`**: PubMed database API integration for term frequency analysis.

### Data Generation (`data_generation/`)

- **`collect_triplets.py`**: UMLS triplet collection and processing.
- **`generate_data.py`**: Primary data generation pipeline based on UMLS triplets.
- **`templates.json`**: Template for question generation.
- **`semantic_to_entity.json`**: Mapping between semantic types and medical entities.
- **`triplets_collection.json`**: Collection of UMLS triplets.

### Data Directory (`data/`)

- **`data.json`**: Sample dataset for medical knowledge evaluation.
- **`hf_dataset/`**: In huggingface dataset format.

## Quick Start

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

## Dataset

Due to UMLS privacy policy restrictions, a sample dataset is provided in `data/data.json` for testing and demonstration purposes. We also have the corresponding huggingface format in `data/hf_dataset/`.

## Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@article{li2025fact,
  title={Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment},
  author={Li, Jiaxi and Wang, Yiwei and Zhang, Kai and Cai, Yujun and Hooi, Bryan and Peng, Nanyun and Chang, Kai-Wei and Lu, Jin},
  journal={arXiv preprint arXiv:2502.14275},
  year={2025}
}
```

