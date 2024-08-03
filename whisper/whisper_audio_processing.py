# This script includes functions for processing audio data, 
# specifically for use in audio-related machine learning tasks. 
"""
!pip install torch transformers librosa ffmpeg-python
# On macOS
brew install ffmpeg
# On Ubuntu
sudo apt-get install ffmpeg
# On Windows
# Download the FFmpeg executable from https://ffmpeg.org/download.html and add it to your PATH.
"""
# screen -S my_session_name
# source /data/daisysxm76/newenv/bin/activate
# Nvidia-smi
# CUDA_VISIBLE_DEVICES=1,2,4,5 python3  /data/daisysxm76/speechtospeech/whisper/whisper/whisper_audio_processing.py  /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/embeddings --num_workers 4
# /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips 

# Detach from the Screen Session:
# Press Ctrl + A followed by D.
# Reattach to the Screen Session (if needed):
# screen -r audio_processing
# Terminate the Screen Session (once the job is complete):
# Type exit within the screen session.

# batching process
# Process the audio files in batches to avoid loading the entire dataset into memory at once.
# Save intermediate results and embeddings to disk in a structured manner.


"""
import os
import argparse
from functools import lru_cache
from typing import Union
import torch.multiprocessing as mp
import logging
import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel
import pdb
import sys 

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.error(f"Failed to load audio {file}: {e.stderr.decode()}")
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

def adjust_shape(input_tensor, target_shape, device):
    input_tensor = input_tensor.cpu().numpy()
    pad_widths = [(0, max(0, target - current)) for target, current in zip(target_shape, input_tensor.shape)]
    adjusted_array = np.pad(input_tensor, pad_widths, mode='constant', constant_values=0)
    adjusted_tensor = torch.tensor(adjusted_array).to(device)
    return adjusted_tensor

def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio).to('cuda')

    if len(audio.shape) > 1:
        audio = audio.squeeze()

    logging.debug(f"Audio tensor shape: {audio.shape}")

    device = audio.device
    window = torch.hann_window(N_FFT).to(device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    logging.debug(f"STFT magnitudes shape: {magnitudes.shape}")

    filters = mel_filters(device, n_mels)

    logging.debug(f"Mel filters shape: {filters.shape}")

    magnitudes = magnitudes.to(device)

    logging.debug(f"Magnitudes shape before adjustment: {magnitudes.shape}")
    logging.debug(f"Target shape for adjustment: {(filters.shape[1], magnitudes.shape[1])}")

    if magnitudes.shape[0] != filters.shape[1]:
        magnitudes = adjust_shape(magnitudes, (filters.shape[1], magnitudes.shape[1]), device)
        logging.debug(f"Magnitudes shape after adjustment: {magnitudes.shape}")

    if filters.shape[1] != magnitudes.shape[0]:
        raise ValueError(f"Shape mismatch: filters.shape[1]={filters.shape[1]}, magnitudes.shape[0]={magnitudes.shape[0]}")

    mel_spec = filters @ magnitudes

    logging.debug(f"Mel spectrogram shape: {mel_spec.shape}")

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def load_whisper_model(model_name: str = "openai/whisper-large-v3"):
    logging.debug(f"Loading Whisper model {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name).to("cuda")
    logging.debug("Model loaded successfully")
    return processor, model

def get_audio_embeddings(audio_file: str, processor, model):
    logging.debug(f"Processing audio file {audio_file}")
    audio = load_audio(audio_file)
    audio = pad_or_trim(audio)
    mel_spec = log_mel_spectrogram(audio)
    inputs = processor(mel_spec.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model.encoder(inputs['input_features']).last_hidden_state
    
    # Copy embeddings to shared memory tensor
    #shared_embedding.copy_(embeddings)

    logging.debug(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def process_file(file, processor, model, output_dir, lock, counter, device):
    logging.info(f"Processing file {file} on {device}")

    progress = (counter.value / 80000) * 100
     # Print progress to console
    sys.stdout.write(f"Processing: {progress:.2f}%\r")
    sys.stdout.flush()
    try:
        embeddings = get_audio_embeddings(file, processor, model)
        output_file = os.path.join(output_dir, os.path.basename(file) + ".pt")
        torch.save(embeddings, output_file)
        logging.info(f"Embeddings saved to {output_file}")

        del embeddings
        torch.cuda.empty_cache()

        with lock:
            counter.value += 1
            progress = (counter.value / 80000) * 100
            sys.stdout.write(f"Processing: {progress:.2f}%\r")
            sys.stdout.flush()
            logging.info(f"Processed {counter.value}/{total_files} files ({progress:.2f}%)")

    except Exception as e:
        logging.error(f"Failed to process {file}: {e}")
        pdb.set_trace()

def main(audio_dir: str, output_dir: str, num_workers: int, devices: list):
    logging.info(f"Starting processing with {num_workers} workers on devices {devices}")

    processor, whisper_model = load_whisper_model()
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]

    if not audio_files:
        logging.error(f"No audio files found in directory {audio_dir}")
        return

    logging.info(f"Found {len(audio_files)} audio files")
    logging.debug(f"Audio files: {audio_files}")

    os.makedirs(output_dir, exist_ok=True)

    manager = mp.Manager()
    lock = manager.Lock()
    counter = manager.Value('i', 0)
    global total_files
    total_files = len(audio_files)

    # Create a shared memory tensor for embeddings
    example_embedding = get_audio_embeddings(audio_files[0], processor, whisper_model)#, torch.zeros(1, 1).share_memory_())
    shared_embedding = torch.zeros_like(example_embedding).share_memory_()
    logging.debug(f"Shared embedding shape: {shared_embedding.shape}")

    pool_args = [(file, processor, whisper_model, output_dir, lock, counter, devices[i % len(devices)]) for i, file in enumerate(audio_files)]

    with mp.Pool(num_workers) as pool:
        pool.starmap(process_file, pool_args)
    print("Processing complete!")
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Initialize CUDA context in the main process
    torch.cuda.init()

    parser = argparse.ArgumentParser(description="Process and save audio embeddings using Whisper in parallel.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files in French")
    parser.add_argument("output_dir", type=str, help="Directory to save the output embeddings")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"], help="List of CUDA devices to use")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir, args.num_workers, args.devices)

"""

import os
import sys
import argparse
from functools import lru_cache
from typing import List
import logging
import ffmpeg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, pipeline
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

# Environment variable settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

class AudioDataset(Dataset):
    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx]  # Return the file path

def load_audio(file: str, sr: int = SAMPLE_RATE):
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logging.error(f"Failed to load audio {file}: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # Convert the byte output to a PyTorch tensor
    audio_array = torch.tensor(np.frombuffer(out, dtype=np.int16).copy()).float().div(32768.0)  # Create a writable copy of the buffer
    return audio_array

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length))
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def adjust_shape(input_tensor, target_shape, device):
    pad_widths = [(0, max(0, target - current)) for target, current in zip(target_shape, input_tensor.shape)]
    adjusted_tensor = F.pad(input_tensor, pad_widths, "constant", 0).to(device)
    return adjusted_tensor

def load_whisper_pipeline(device='cuda:0'):
    model_id = "openai/whisper-large-v3"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    processor = WhisperProcessor.from_pretrained(model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return whisper_pipe, processor, model

def get_audio_embeddings(audio_file: str, whisper_pipe, processor):
    logging.debug(f"Processing audio file {audio_file}")
    audio = load_audio(audio_file)
    audio = pad_or_trim(audio)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(whisper_pipe.device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = whisper_pipe.model(**inputs).last_hidden_state

    logging.debug(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def project_embeddings(embeddings, projection_matrix):
    logging.debug("Projecting embeddings to match LLaMA input dimensions")
    return torch.matmul(embeddings, projection_matrix)

def fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, device, scaler):
    # Ensure the model has not been configured with multiple adapters
    if not hasattr(llama_model, 'peft_config'):
        llama_model = get_peft_model(llama_model, lora_config)

    llama_model.train()

    embeddings = embeddings.to(device)
    with autocast():
        outputs = llama_model(inputs_embeds=embeddings)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

def process_file(audio_file: str, whisper_pipe, processor, llama_model, accelerator, lora_config, optimizer, llama_device, scaler, projection_matrix):
    logging.info(f"Processing audio file {audio_file}")
    try:
        embeddings = get_audio_embeddings(audio_file, whisper_pipe, processor)
        embeddings = project_embeddings(embeddings, projection_matrix)
        fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while processing audio. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                embeddings = get_audio_embeddings(audio_file, whisper_pipe, processor)
                embeddings = project_embeddings(embeddings, projection_matrix)
                fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA out of memory again while processing audio. Exiting.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to process audio: {e}")
                    sys.exit(1)
        else:
            logging.error(f"Failed to process audio: {e}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to process audio: {e}")
        sys.exit(1)

def save_fine_tuned_lora_weights(llama_model, output_dir):
    logging.info(f"Saving fine-tuned LoRA weights to {output_dir}")
    lora_weights_path = os.path.join(output_dir, "lora_weights.bin")
    torch.save(llama_model.state_dict(), lora_weights_path)
    logging.info("LoRA weights saved successfully")

def load_data_from_tsv(tsv_file: str, audio_dir: str, max_samples: int = 1000):
    df = pd.read_csv(tsv_file, sep='\t')

    # Select the first max_samples entries
    df = df.head(max_samples)

    audio_files = df['path'].tolist()

    # Filter out missing files
    valid_audio_files = []
    for audio_file in audio_files:
        full_path = os.path.join(audio_dir, audio_file)
        if os.path.exists(full_path):
            valid_audio_files.append(full_path)
        else:
            logging.error(f"Audio file {full_path} does not exist.")
    
    return valid_audio_files

def main(audio_dir: str, output_dir: str, whisper_device: str, llama_device: str, tsv_file: str):
    logging.info(f"Starting processing on devices whisper: {whisper_device}, llama: {llama_device}")

    # Ensure the specified CUDA devices are valid
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Please check your CUDA installation.")
        sys.exit(1)

    available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available CUDA devices: {available_devices}")

    try:
        whisper_pipe, processor, whisper_model = load_whisper_pipeline(device=whisper_device)
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)

    model_id = "meta-llama/Meta-Llama-3-8B"

    try:
        llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        llama_model.to(llama_device)
    except Exception as e:
        logging.error(f"Failed to load Llama model: {e}")
        sys.exit(1)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )

    # Create a projection matrix to match the dimensions
    whisper_embedding_dim = whisper_model.config.hidden_size
    llama_embedding_dim = llama_model.config.hidden_size
    projection_matrix = torch.nn.Linear(whisper_embedding_dim, llama_embedding_dim).to(llama_device)  # Ensure correct device for projection

    accelerator = Accelerator(device_placement=False)  # Disable automatic device placement by Accelerator
    optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Prepare the model, optimizer, and possibly dataloaders
    llama_model, optimizer = accelerator.prepare(llama_model, optimizer)

    audio_files = load_data_from_tsv(tsv_file, audio_dir)

    if not audio_files:
        logging.error(f"No audio files found in the TSV file")
        return

    logging.info(f"Found {len(audio_files)} valid audio files")
    logging.debug(f"Valid audio files: {audio_files}")

    dataset = AudioDataset(audio_files)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    for batch in dataloader:
        audio_file = batch[0]  # Get the file path
        process_file(audio_file, whisper_pipe, processor, llama_model, accelerator, lora_config, optimizer, llama_device, scaler, projection_matrix)

    print("Processing complete!")

    # Save the fine-tuned LoRA weights
    save_fine_tuned_lora_weights(llama_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
    parser.add_argument("--whisper_device", type=str, default="cuda:0", help="CUDA device to use for Whisper model")
    parser.add_argument("--llama_device", type=str, default="cuda:0", help="CUDA device to use for LLaMA model")
    parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir, args.whisper_device, args.llama_device, args.tsv_file)


