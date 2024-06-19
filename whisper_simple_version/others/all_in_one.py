# process_data.py
import pandas as pd
import os
from transcribe import transcribe_audio
from translate import translate_text
from tqdm import tqdm
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

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
            logging.info(f"Translated Text: {translated_text}")
            return transcribed_text, translated_text
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

    # Process files sequentially
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        try:
            transcribed_text, translated_text = process_file(row, args.data_dir)
            if transcribed_text and translated_text:
                df.at[index, "transcribed_text"] = transcribed_text
                df.at[index, "translated_text"] = translated_text
        except Exception as exc:
            logging.error(f"Generated an exception: {exc}")

    # Save the updated DataFrame to a new CSV file
    df.to_csv(args.output_file, sep='\t', index=False)
    logging.info(f"Processed dataset saved to {args.output_file}")

# transcribe.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import resampy
import soundfile as sf
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")

def transcribe_audio(file_path):
    try:
        # Load audio file without specifying additional parameters
        audio_input, sample_rate = sf.read(file_path)

        # Resample to 16000 Hz if the sample rate is different
        if sample_rate != 16000:
            audio_input = resampy.resample(audio_input, sample_rate, 16000)
            sample_rate = 16000

        # Preprocess the audio input
        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to("cuda")

        # Perform inference
        generated_ids = model.generate(input_features)

        # Decode the transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

# translate.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat")

def translate_text(text, source_lang="fr", target_lang="en"):
    try:
        # Prepare the input text with language tokens if required by the model
        inputs = tokenizer(f"<s> {source_lang} {text} </s> {target_lang}", return_tensors="pt")
        outputs = model.generate(**inputs)
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translation[0]
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text using a translation model")
    parser.add_argument("text", type=str, help="Text to translate")
    parser.add_argument("--source_lang", type=str, default="fr", help="Source language (default: 'fr')")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (default: 'en')")
    args = parser.parse_args()

    result = translate_text(args.text, args.source_lang, args.target_lang)
    if result:
        logging.info(f"Translated Text: {result}")
    else:
        logging.warning("Failed to translate text.")
