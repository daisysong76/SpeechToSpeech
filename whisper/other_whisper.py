import os
import argparse
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

def load_audio(file: str, sr: int = SAMPLE_RATE):
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def load_whisper_model(model_name: str = "openai/whisper-large-v3"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    return processor, model

def get_audio_embeddings(audio_file: str, processor, model):
    audio = load_audio(audio_file)
    audio = pad_or_trim(audio)
    mel_spec = log_mel_spectrogram(audio)
    inputs = processor(mel_spec, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        embeddings = model.encoder(inputs.input_features).last_hidden_state

    return embeddings

def main(audio_file: str, output_file: str):
    processor, whisper_model = load_whisper_model()
    embeddings = get_audio_embeddings(audio_file, processor, whisper_model)
    
    torch.save(embeddings, output_file)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save audio embeddings using Whisper.")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file in French")
    parser.add_argument("output_file", type=str, help="Path to save the output embeddings")
    args = parser.parse_args()

    main(args.audio_file, args.output_file)
