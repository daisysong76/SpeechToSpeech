# process_data.py
import pandas as pd
import os
from transcribe import transcribe_audio
from speechtospeech.whisper.whisper_simple_version.others.translate_Helsinki import translate_text
from tqdm import tqdm

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

def process_file(row, data_dir):
    audio_column = 'path'  # Replace 'path' with the actual column name for audio file paths
    audio_file_path = os.path.join(data_dir, row[audio_column])
    if os.path.exists(audio_file_path):
        print(f"Processing file: {audio_file_path}")
        transcribed_text = transcribe_audio(audio_file_path)
        if transcribed_text:
            print(f"Transcribed Text: {transcribed_text}")
            translated_text = translate_text(transcribed_text)
            print(f"Translated Text: {translated_text}")
            return transcribed_text, translated_text
        else:
            print(f"Failed to transcribe {audio_file_path}")
            return None, None
    else:
        print(f"Audio file not found: {audio_file_path}")
        return None, None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("output_file", type=str, help="Path to save the processed dataset CSV file")
    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"Dataset loaded. Number of rows: {len(df)}")

    # Initialize columns for transcriptions and translations
    df["transcribed_text"] = None
    df["translated_text"] = None

    # Process files sequentially
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        try:
            transcribed_text, translated_text = process_file(row, args.data_dir)
            if transcribed_text and translated_text:
                df.at[index, "transcribed_text"] = transcribed_text
                df.at[index, "translated_text"] = translated_text
        except Exception as exc:
            print(f"Generated an exception: {exc}")

    # Save the updated DataFrame to a new CSV file
    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Processed dataset saved to {args.output_file}")

