import pandas as pd
import os
from transcribe import transcribe_audio
from translate0 import translate_text
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
            translated_text = translate_text(transcribed_text)
            if translated_text:
                logging.info(f"Translated Text: {translated_text}")
                # Extract only the English part
                start_token = "</s>en>"
                end_token = "</s>"
                if start_token in translated_text:
                    english_part = translated_text.split(start_token)[-1].split(end_token)[0].strip()
                else:
                    english_part = translated_text  # Default to full text if tokens are not found
                logging.info(f"Translated Text (English): {english_part}")
                return transcribed_text, english_part
            else:
                logging.warning(f"Failed to translate {transcribed_text}")
                return transcribed_text, None
        else:
            logging.warning(f"Failed to transcribe {audio_file_path}")
            return None, None
    else:
        logging.warning(f"Audio file not found: {audio_file_path}")
        return None, None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("output_file", type=str, help="Path to save the processed dataset CSV file")
    args = parser.parse_args()

    # Load the dataset
    logging.info("Loading dataset...")
    df = load_dataset(args.dataset)
    logging.info(f"Dataset loaded. Number of rows: {len(df)}")

    # Initialize columns for transcriptions and translations
    df["transcribed_text"] = None
    df["translated_text"] = None

    # Process each file sequentially
    for index, row in df.iterrows():
        transcribed_text, translated_text = process_file(row, args.data_dir)
        if transcribed_text is not None:
            df.at[index, "transcribed_text"] = transcribed_text
        if translated_text is not None:
            df.at[index, "translated_text"] = translated_text

    # Save the updated DataFrame to a new CSV file
    df.to_csv(args.output_file, sep='\t', index=False)
    logging.info(f"Processed dataset saved to {args.output_file}")


