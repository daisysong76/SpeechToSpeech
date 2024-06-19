import pandas as pd
import os
from transcribe import transcribe_audio
from speechtospeech.whisper.whisper_simple_version.others.translate_Helsinki import translate_text

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.dataset)
    print(df.columns)  # Print column names to verify

    # Update the column name here based on the actual column name in your TSV file
    audio_column = 'path'  # Replace 'path' with the actual column name for audio file paths

    # Process each audio file in the dataset
    for index, row in df.iterrows():
        audio_file_path = os.path.join(args.data_dir, row[audio_column])
        if os.path.exists(audio_file_path):
            # Transcribe the audio to text
            transcribed_text = transcribe_audio(audio_file_path)
            print(f"Transcribed Text: {transcribed_text}")

            # Translate the transcribed text from French to English
            translated_text = translate_text(transcribed_text)
            print(f"Translated Text: {translated_text}")
        else:
            print(f"Audio file not found: {audio_file_path}")


"""
Update the Main Script to Handle Multiple Languages
# main.py
import pandas as pd
import os
from transcribe import transcribe_audio
from translate import translate_text

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio to multiple languages")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("--target_langs", nargs="+", required=True, help="List of target languages for translation")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.dataset)
    print(df.columns)  # Print column names to verify

    # Update the column name here based on the actual column name in your TSV file
    audio_column = 'path'  # Replace 'path' with the actual column name for audio file paths

    # Process each audio file in the dataset
    for index, row in df.iterrows():
        audio_file_path = os.path.join(args.data_dir, row[audio_column])
        if os.path.exists(audio_file_path):
            try:
                # Transcribe the audio to text
                transcribed_text = transcribe_audio(audio_file_path)
                print(f"Transcribed Text: {transcribed_text}")

                # Translate the transcribed text to multiple target languages
                for target_lang in args.target_langs:
                    translated_text = translate_text(transcribed_text, source_lang="fr", target_lang=target_lang)
                    print(f"Translated Text ({target_lang}): {translated_text}")
            except Exception as e:
                print(f"Error processing {audio_file_path}: {e}")
        else:
            print(f"Audio file not found: {audio_file_path}")
"""
"""
import pandas as pd
import os
from transcribe import transcribe_audio
from translate import translate_text

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.dataset)

    # Process each audio file in the dataset
    for index, row in df.iterrows():
        audio_file_path = os.path.join(args.data_dir, row['audio'])
        if os.path.exists(audio_file_path):
            # Transcribe the audio to text
            transcribed_text = transcribe_audio(audio_file_path)
            print(f"Transcribed Text: {transcribed_text}")

            # Translate the transcribed text from French to English
            translated_text = translate_text(transcribed_text)
            print(f"Translated Text: {translated_text}")
        else:
            print(f"Audio file not found: {audio_file_path}")
"""

"""import pandas as pd
import os
from transcribe import transcribe_audio
from translate import translate_text

def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and translate audio from French to English")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing audio files")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.dataset)

    # Process each audio file in the dataset
    for index, row in df.iterrows():
        audio_file_path = os.path.join(args.data_dir, row['audio'])
        if os.path.exists(audio_file_path):
            # Transcribe the audio to text
            transcribed_text = transcribe_audio(audio_file_path)
            print(f"Transcribed Text: {transcribed_text}")

            # Translate the transcribed text from French to English
            translated_text = translate_text(transcribed_text)
            print(f"Translated Text: {translated_text}")
        else:
            print(f"Audio file not found: {audio_file_path}")
            """

