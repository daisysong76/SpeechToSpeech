"""
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os

# Download necessary NLTK data files
nltk.download('punkt')

def process_translations_corpus_bleu(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall BLEU score calculation
    references = []
    hypotheses = []

    # Define smoothing function
    smoothie = SmoothingFunction().method4

    # Prepare data for corpus BLEU calculation
    for item in data:
        reference = item["tsv_translation"]
        hypothesis = item["translated_text"]

        # Tokenize the reference and hypothesis
        reference_tokens = [nltk.word_tokenize(reference)]
        hypothesis_tokens = nltk.word_tokenize(hypothesis)

        references.append(reference_tokens)
        hypotheses.append(hypothesis_tokens)

    # Calculate corpus BLEU score
    corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

    # Create the final output dictionary
    output_data = {
        "corpus_bleu_score": corpus_bleu_score
    }

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the BLEU scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Corpus BLEU score has been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/matched_results.json'
output_file = '/data/daisysxm76/speechtospeech/output_corpus_BLEU.json'

# Process the translations and calculate BLEU scores
process_translations_corpus_bleu(input_file, output_file)
"""
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os

# Download necessary NLTK data files
nltk.download('punkt')

# Include the normalize_text function from above here
from normalize_text import normalize_text

def process_translations_corpus_bleu(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall BLEU score calculation
    references = []
    hypotheses = []

    # Define smoothing function
    smoothie = SmoothingFunction().method4

    # Prepare data for corpus BLEU calculation
    for item in data:
        reference = item["tsv_translation"]
        hypothesis = item["translated_text"]

        # Normalize the reference and hypothesis
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

        # Tokenize the reference and hypothesis
        reference_tokens = [nltk.word_tokenize(reference)]
        hypothesis_tokens = nltk.word_tokenize(hypothesis)

        references.append(reference_tokens)
        hypotheses.append(hypothesis_tokens)

    # Calculate corpus BLEU score
    corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

    # Create the final output dictionary
    output_data = {
        "corpus_bleu_score": corpus_bleu_score
    }

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the BLEU scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Corpus BLEU score has been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/matched_results.json'
output_file = '/data/daisysxm76/speechtospeech/output_corpus_BLEU.json'

# Process the translations and calculate BLEU scores
process_translations_corpus_bleu(input_file, output_file)
