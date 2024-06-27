"""
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

# Download necessary NLTK data files
nltk.download('punkt')

def process_translations(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall BLEU score calculation
    bleu_scores = []
    total_bleu_score = 0
    count = 0

    # Define smoothing function
    smoothie = SmoothingFunction().method4

    # Calculate BLEU for each item
    for item in data:
        reference = item["tsv_translation"]
        hypothesis = item["translated_text"]

        # Tokenize the reference and hypothesis
        reference_tokens = [nltk.word_tokenize(reference)]
        hypothesis_tokens = nltk.word_tokenize(hypothesis)

        # Calculate BLEU score
        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
        total_bleu_score += bleu_score
        count += 1

        bleu_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "translated_text": hypothesis,
            "reference_text": reference,
            "bleu_score": bleu_score
        })

    # Calculate overall average BLEU score
    overall_bleu_score = total_bleu_score / count if count > 0 else 0

    # Create the final output dictionary
    output_data = {
        "individual_bleu_scores": bleu_scores,
        "overall_bleu_score": overall_bleu_score
    }

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the BLEU scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"BLEU scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/matched_results.json'
output_file = '/data/daisysxm76/speechtospeech/output_BLEU.json'

# Process the translations and calculate BLEU scores
process_translations(input_file, output_file)
"""

import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

# Download necessary NLTK data files
nltk.download('punkt')

# Include the normalize_text function from above here
from normalize_text import normalize_text

def process_translations_sentence_bleu(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall BLEU score calculation
    total_bleu_score = 0
    count = 0
    bleu_scores = []

    # Define smoothing function
    smoothie = SmoothingFunction().method4

    # Calculate BLEU for each item
    for item in data:
        reference = item["tsv_translation"]
        hypothesis = item["translated_text"]

        # Normalize the reference and hypothesis
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

        # Tokenize the reference and hypothesis
        reference_tokens = [nltk.word_tokenize(reference)]
        hypothesis_tokens = nltk.word_tokenize(hypothesis)

        # Calculate BLEU score
        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
        total_bleu_score += bleu_score
        count += 1

        bleu_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "translated_text": hypothesis,
            "reference_text": reference,
            "bleu_score": bleu_score
        })

    # Calculate overall average BLEU score
    overall_bleu_score = total_bleu_score / count if count > 0 else 0

    # Create the final output dictionary
    output_data = {
        "individual_bleu_scores": bleu_scores,
        "overall_bleu_score": overall_bleu_score
    }

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the BLEU scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"BLEU scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/matched_results.json'
output_file = '/data/daisysxm76/speechtospeech/output_sentence_BLEU.json'

# Process the translations and calculate BLEU scores
process_translations_sentence_bleu(input_file, output_file)
