import os
import argparse
import logging
import numpy as np
import torch
import ffmpeg
from transformers import WhisperProcessor, WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE

def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file using ffmpeg, convert to mono and resample.
    """
    try:
        out, _ = (
            ffmpeg.input(file_path)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=str(sample_rate))
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768
    except ffmpeg.Error as e:
        logging.error(f"Failed to load audio {file_path}: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return audio

def adjust_shape(input_tensor):
    expected_shape = (3, 2)
    # Check if the input tensor shape matches the expected shape
    if input_tensor.shape != expected_shape:
        # Move tensor to CPU
        input_tensor = input_tensor.cpu()
        # Convert tensor to NumPy array
        input_array = input_tensor.numpy()
        # Calculate padding needed to match the expected shape
        padding = [(0, max(0, expected_dim - actual_dim)) for expected_dim, actual_dim in zip(expected_shape, input_array.shape)]
        # Apply padding
        adjusted_array = np.pad(input_array, padding, mode='constant', constant_values=0)
        # Convert NumPy array back to tensor and move to CUDA
        adjusted_tensor = torch.tensor(adjusted_array).to('cuda')
        return adjusted_tensor
    else:
        return input_tensor

def get_audio_embeddings(audio_file: str, processor, model) -> torch.Tensor:
    """
    Generate embeddings for an audio file using Whisper model.
    """
    audio = load_audio(audio_file)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to("cuda")
    with torch.no_grad():
        embeddings = model.encoder(inputs).last_hidden_state
        embeddings = adjust_shape(embeddings)
    return embeddings

def process_file(file, processor, model, output_dir, counter, total_files):
    """
    Process a single audio file to extract embeddings and save them.
    """
    try:
        embeddings = get_audio_embeddings(file, processor, model)
        output_file = os.path.join(output_dir, os.path.basename(file) + ".pt")
        torch.save(embeddings, output_file)
        counter[0] += 1
        progress = (counter[0] / total_files) * 100
        logging.info(f"Processed {counter[0]}/{total_files} files ({progress:.2f}%)")
    except Exception as e:
        logging.error(f"Failed to process {file}: {e}", exc_info=True)

def main(audio_dir: str, output_dir: str):
    """
    Main function to process all audio files in a directory.
    """
    logging.info("Starting processing")

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperModel.from_pretrained("openai/whisper-large-v2").to("cuda")
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]

    if not audio_files:
        logging.error("No audio files found in the specified directory.")
        return

    os.makedirs(output_dir, exist_ok=True)

    total_files = len(audio_files)
    counter = [0]

    for file in audio_files:
        process_file(file, processor, model, output_dir, counter, total_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files and extract embeddings using Whisper model.")
    parser.add_argument("audio_dir", help="Directory containing the audio files")
    parser.add_argument("output_dir", help="Directory to save the audio embeddings")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir)
