from Bio import Entrez
import time
from tqdm import tqdm
import utils


Entrez.email = "your-registration-email"
Entrez.api_key = "your-api-key"


def search_term_freq_pubmed(term):
    # Retrieve the number of articles containing the term
    handle = Entrez.esearch(db="pubmed", term=term, retmode="xml")
    record = Entrez.read(handle)
    # Get the number of search results
    cnt = int(record["Count"])
    time.sleep(0.1)
    return cnt

def get_term_frequencies():
    data = utils.load_json("../data_generation/data.json")
    # Extract all medical terms from triplets
    medical_terms = set()
    semantic_type = {}
    for item in data:
        triplet = item['original_triplet']
        pos_statement = item['pos_qa']['statement']
        pos_object = triplet['object']
        medical_terms.add(pos_object)
        semantic_type[pos_object] = item['semantic_type']
        # Determine the template by replacing the pos_object with a placeholder
        template = pos_statement.replace(pos_object, "{object}")
        for neg_statement in item['neg_qa']['statement']:
            neg_words = []
            for word in neg_statement.split():
                if word not in template:
                    neg_words.append(word)
            
            if neg_words:
                neg_object = ' '.join(neg_words)
                medical_terms.add(neg_object)
                semantic_type[neg_object] = item['semantic_type']
            else:
                raise ValueError(f"No neg_object found for {neg_statement}")

    # Search frequency for each term and store results
    term_frequencies = []
    for term in tqdm(medical_terms):
        try:
            freq = search_term_freq_pubmed(term)
            term_frequencies.append({"term": term, 'semantic_type': semantic_type[term], "frequency": freq})
            print(f"Term '{term}' appears {freq} times in PubMed")
        except Exception as e:
            print(f"Failed to get frequency for term: {term}")
            print(e)
            continue

    utils.write_json("../data_generation/term_frequencies_pubmed.json", term_frequencies)
    print(f"\nSaved frequencies for {len(term_frequencies)} terms to term_frequencies.json")

    # Calculate average frequency for each semantic type
    semantic_type_freqs = {}
    semantic_type_counts = {}
    for item in term_frequencies:
        for sem_type in item['semantic_type']:
            if sem_type not in semantic_type_freqs:
                semantic_type_freqs[sem_type] = 0
                semantic_type_counts[sem_type] = 0
            semantic_type_freqs[sem_type] += item['frequency']
            semantic_type_counts[sem_type] += 1
    
    print("\nAverage frequencies by semantic type:")
    avg_frequencies = []
    for sem_type in semantic_type_freqs:
        avg_freq = semantic_type_freqs[sem_type] / semantic_type_counts[sem_type]
        avg_frequencies.append((sem_type, avg_freq))
    avg_frequencies.sort(key=lambda x: x[1])    
    for sem_type, avg_freq in avg_frequencies:
        print(f"{sem_type}: {avg_freq:.2f}")


if __name__ == "__main__":
    get_term_frequencies()
