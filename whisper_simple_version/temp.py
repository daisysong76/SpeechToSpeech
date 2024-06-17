
# process_data.py
import pandas as pd
from transcribe import transcribe_audio
from speechtospeech.whisper.whisper_simple_version.translate_Helsinki import translate_text
import os
from tqdm import tqdm

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.dataset)
    print(f"Dataset loaded. Number of rows: {len(df)}")

    # Update the column name here based on the actual column name in your TSV file
    audio_column = 'path'  # Replace 'path' with the actual column name for audio file paths

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        audio_file_path = os.path.join(args.data_dir, row[audio_column])
        if os.path.exists(audio_file_path):
            # Transcribe the audio to text
            transcribed_text = transcribe_audio(audio_file_path)
            print(f"Transcribed Text: {transcribed_text}")

            # Translate the transcribed text from French to English
            translated_text = translate_text(transcribed_text)
            print(f"Original transcription: {transcribed_text}")
            print(f"Translated Text: {translated_text}")
            # Here, you would also have the target translation for comparison
            # For this example, I'm assuming it's provided in another column in the TSV
            target_translation = row['target_translation']  # Replace with actual column name if available
            print(f"Target Translation: {target_translation}")
        else:
            print(f"Audio file not found: {audio_file_path}")


