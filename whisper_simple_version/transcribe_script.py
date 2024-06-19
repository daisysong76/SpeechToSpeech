import pandas as pd
import os
import json
from transcribe import transcribe_audio
from tqdm import tqdm
import logging
from utils.logging_config import setup_logging

# Initialize logging
setup_logging()

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

def process_file(row, data_dir):
    audio_column = 'path'  # Replace 'path' with the actual column name for audio file paths
    audio_file_path = os.path.join(data_dir, row[audio_column])
    if os.path.exists(audio_file_path):
        logging.info(f"Processing file: {audio_file_path}")
        transcribed_text = transcribe_audio(audio_file_path)
        if transcribed_text:
            logging.info(f"Transcribed Text: {transcribed_text}")
            return transcribed_text
        else:
            logging.warning(f"Failed to transcribe {audio_file_path}")
            return None
    else:
        logging.warning(f"Audio file not found: {audio_file_path}")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio to text using Whisper model")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("output_file", type=str, help="Path to save the transcriptions JSON file")
    args = parser.parse_args()

    # Load the dataset
    logging.info("Loading dataset...")
    df = load_dataset(args.dataset)
    logging.info(f"Dataset loaded. Number of rows: {len(df)}")

    # Initialize list for results
    results = []

    # Process files sequentially
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        transcribed_text = process_file(row, args.data_dir)
        if transcribed_text is not None:
            result = {
                "index": index,
                "transcribed_text": transcribed_text
            }
            results.append(result)

    # Save the results to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Transcriptions saved to {args.output_file}")
