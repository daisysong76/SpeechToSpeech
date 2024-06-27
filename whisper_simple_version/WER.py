

"""
import json
import numpy as np
import os

def calculate_wer(reference, hypothesis):
    # Split the reference and hypothesis into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Create a matrix to store the edit distances
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

    # Initialize the matrix
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # The WER is the edit distance divided by the number of words in the reference
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer, d[len(ref_words)][len(hyp_words)]

# Function to process the dataset and calculate WER scores
def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        wer, errors = calculate_wer(original_transcription, transcribed_text)
        total_errors += errors
        total_words += len(original_transcription.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Print statements for debugging
    print(f"Total Errors: {total_errors}")
    print(f"Total Words: {total_words}")
    print(f"Overall WER: {overall_wer}")

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = 'path_transcriptions.json'
output_file = 'wer_score.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)

"""
"""
import json
import os
from wer import wer

def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        wer_score = calculate_wer(original_transcription, transcribed_text)
        total_errors += wer_score * len(original_transcription.split())
        total_words += len(original_transcription.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer_score
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = 'path_transcriptions.json'
output_file = 'wer_score.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)

"""
"""
import json
import jiwer
import os


# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
])


def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        wer = jiwer.wer(original_transcription, transcribed_text)
        total_errors += wer * len(original_transcription.split())
        total_words += len(original_transcription.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")


input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)

"""
"""
import json
import jiwer
import os

# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
])

def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        
        # Apply the transformation to both reference and hypothesis
        transformed_original = transformation(original_transcription)
        transformed_transcribed = transformation(transcribed_text)
        
        wer = jiwer.wer(transformed_original, transformed_transcribed)
        total_errors += wer * len(transformed_original.split())
        total_words += len(transformed_original.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)

"""
"""
# This is the result after handling Unicode and Accented Characters: "overall_wer": 0.11177976463395996
import json
import jiwer
import os
import string
import unicodedata

# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
])

def normalize_text(text):

    # Convert to lowercase
    text = text.lower()
    # Normalize unicode characters (e.g., accented characters)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespaces
    text = text.strip()
    # Replace all types of whitespace characters with a single space
    text = ' '.join(text.split())
    return text

def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        
        # Apply the transformation to both reference and hypothesis
        transformed_original = transformation(normalize_text(original_transcription))
        transformed_transcribed = transformation(normalize_text(transcribed_text))
        
        wer = jiwer.wer(transformed_original, transformed_transcribed)
        total_errors += wer * len(transformed_original.split())
        total_words += len(transformed_original.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)
"""


"""

import json
import numpy as np
import os
import string

def normalize_text(text):
    
    #Normalize the text by converting to lowercase, removing punctuation,
    #and handling whitespace.
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespaces
    text = text.strip()
    # Replace all types of whitespace characters with a single space
    text = ' '.join(text.split())
    return text

def calculate_wer(reference, hypothesis):
    # Normalize the texts
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    
    # Split the reference and hypothesis into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

     # Use SequenceMatcher to find the best alignment
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    opcodes = matcher.get_opcodes()

    substitutions = sum(1 for tag, i1, i2, j1, j2 in opcodes if tag == 'replace')
    deletions = sum(1 for tag, i1, i2, j1, j2 in opcodes if tag == 'delete')
    insertions = sum(1 for tag, i1, i2, j1, j2 in opcodes if tag == 'insert')

    wer = (substitutions + deletions + insertions) / len(ref_words)

    # Create a matrix to store the edit distances
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

    # Initialize the matrix
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # The WER is the edit distance divided by the number of words in the reference
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer, d[len(ref_words)][len(hyp_words)]

# Function to process the dataset and calculate WER scores
def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        wer, errors = calculate_wer(original_transcription, transcribed_text)
        total_errors += errors
        total_words += len(original_transcription.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Print statements for debugging
    print(f"Total Errors: {total_errors}")
    print(f"Total Words: {total_words}")
    print(f"Overall WER: {overall_wer}")

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)

"""
"""
import json
import jiwer
import os
import string
import unicodedata

# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
])

def normalize_text(text):
   
    # Convert to lowercase
    text = text.lower()
    # Normalize unicode characters (e.g., accented characters)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespaces
    text = text.strip()
    # Replace all types of whitespace characters with a single space
    text = ' '.join(text.split())
    return text

def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        
        # Apply the transformation to both reference and hypothesis
        transformed_original = transformation(normalize_text(original_transcription))
        transformed_transcribed = transformation(normalize_text(transcribed_text))
        
        wer = jiwer.wer(transformed_original, transformed_transcribed)
        total_errors += wer * len(transformed_original.split())
        total_words += len(transformed_original.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)
"""

import json
import jiwer
import os
import string
import unicodedata
import inflect

# Initialize inflect engine for number to word conversion
p = inflect.engine()

# Dictionary for contractions and abbreviations
contractions_dict = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'ll": " will",
    "'ve": " have",
    "'re": " are",
    "'d": " would",
    "it's": "it is",
    "i'm": "i am",
    "let's": "let us",
    # Add more as needed
}

def expand_contractions(text):
    for contraction, full_form in contractions_dict.items():
        text = text.replace(contraction, full_form)
    return text

def normalize_numbers(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if word.isdigit():
            word = p.number_to_words(word)
        normalized_words.append(word)
    return ' '.join(normalized_words)

def normalize_text(text):
    """
    Normalize the text by converting to lowercase, removing punctuation,
    handling whitespace, normalizing unicode characters, expanding contractions,
    and converting numbers to words.
    """
    # Convert to lowercase
    text = text.lower()
    # Normalize unicode characters (e.g., accented characters)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespaces
    text = text.strip()
    # Replace all types of whitespace characters with a single space
    text = ' '.join(text.split())
    # Expand contractions and abbreviations
    text = expand_contractions(text)
    # Normalize numbers
    text = normalize_numbers(text)
    return text

# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
])

def process_transcriptions(input_path, output_path):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input dataset
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize variables for overall WER calculation
    total_errors = 0
    total_words = 0

    # Calculate WER for each item
    wer_scores = []
    for item in data:
        transcribed_text = item["transcribed_text"]
        original_transcription = item["original_transcription"]
        
        # Apply the transformation to both reference and hypothesis
        transformed_original = transformation(normalize_text(original_transcription))
        transformed_transcribed = transformation(normalize_text(transcribed_text))
        
        wer = jiwer.wer(transformed_original, transformed_transcribed)
        total_errors += wer * len(transformed_original.split())
        total_words += len(transformed_original.split())
        wer_scores.append({
            "index": item.get("index"),
            "audio_file": item.get("audio_file"),
            "transcribed_text": transcribed_text,
            "original_transcription": original_transcription,
            "wer_score": wer
        })
    
    # Calculate overall WER
    overall_wer = total_errors / total_words

    # Create the final output dictionary
    output_data = {
        "individual_wer_scores": wer_scores,
        "overall_wer": overall_wer
    }
  
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the WER scores to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"WER scores have been calculated and saved to {output_path}")

# Define input and output file paths
input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/path_transcriptions.json'
output_file = '/data/daisysxm76/speechtospeech/output_WER.json'

# Process the transcriptions and calculate WER scores
process_transcriptions(input_file, output_file)
