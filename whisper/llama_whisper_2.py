import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from functools import lru_cache
from typing import Union, List
import logging
import ffmpeg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import sys

from models import TLTR
tltr_model = TLTR()  # Initialize TLTR model
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
        audio_file = self.audio_files[idx]
        if not os.path.exists(audio_file):
            logging.error(f"Audio file {audio_file} does not exist. Skipping this file.")
            return None
        audio = load_audio(audio_file)
        audio = pad_or_trim(audio)
        return audio

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

    audio_tensor = torch.frombuffer(out, dtype=torch.int16).float() / 32768.0
    return audio_tensor

def pad_or_trim(tensor, length: int = N_SAMPLES, *, axis: int = -1):
    if tensor.shape[axis] > length:
        tensor = tensor.index_select(dim=axis, index=torch.arange(length, device=tensor.device))
    if tensor.shape[axis] < length:
        pad_widths = [(0, 0)] * tensor.ndim
        pad_widths[axis] = (0, length - tensor.shape[axis])
        tensor = F.pad(tensor, [pad for sizes in pad_widths[::-1] for pad in sizes])
    return tensor

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    filters = torch.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.pt"))
    return filters[f"mel_{n_mels}"].to(device)

def adjust_shape(input_tensor, target_shape, device):
    pad_widths = [(0, max(0, target - current)) for target, current in zip(target_shape, input_tensor.shape)]
    adjusted_tensor = F.pad(input_tensor, pad_widths)
    return adjusted_tensor.to(device)

def log_mel_spectrogram(audio: Union[str, torch.Tensor], n_mels: int = N_MELS, device='cuda'):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.tensor(audio).to(device)

    if len(audio.shape) > 1:
        audio = audio.squeeze()

    logging.debug(f"Audio tensor shape: {audio.shape}")

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

def load_whisper_model(device='cuda:0'):
    model_name = "openai/whisper-large-v3"
    logging.debug(f"Loading Whisper model {model_name}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperModel.from_pretrained(model_name).to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA out of memory again while loading Whisper model. Exiting.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to load Whisper model: {e}")
                    sys.exit(1)
        else:
            logging.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    logging.debug("Model loaded successfully")
    return processor, model

def get_audio_embeddings(audio, processor, model, device='cuda'):
    mel_spec = log_mel_spectrogram(audio, device=device)
    inputs = processor(mel_spec.cpu(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model.encoder(inputs['input_features']).last_hidden_state

    logging.debug(f"Embeddings shape: {embeddings.shape}")
    return embeddings

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

def process_file(audio, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler):
    logging.info(f"Processing audio")
    try:
        embeddings = get_audio_embeddings(audio, processor, whisper_model, device=accelerator.device)
        fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while processing audio. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                embeddings = get_audio_embeddings(audio, processor, whisper_model, device=accelerator.device)
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
    df = df.head(max_samples)
    audio_files = [os.path.join(audio_dir, file) for file in df['path'].tolist()]
    return audio_files

def main(audio_dir: str, output_dir: str, device: str, tsv_file: str):
    logging.info(f"Starting processing on device {device}")

    # Ensure the specified CUDA devices are valid
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Please check your CUDA installation.")
        sys.exit(1)

    available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available CUDA devices: {available_devices}")

    try:
        processor, whisper_model = load_whisper_model(device=device)
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)

    model_id = "meta-llama/Meta-Llama-3-8B"

    try:
        llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        llama_model.to(device)
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

    accelerator = Accelerator(device_placement=False)  # Disable automatic device placement by Accelerator
    optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Prepare the model, optimizer, and possibly dataloaders
    llama_model, optimizer = accelerator.prepare(llama_model, optimizer)

    audio_files = load_data_from_tsv(tsv_file, audio_dir)

    if not audio_files:
        logging.error(f"No audio files found in the TSV file")
        return

    logging.info(f"Found {len(audio_files)} audio files")
    logging.debug(f"Audio files: {audio_files}")

    dataset = AudioDataset(audio_files)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    for batch in dataloader:
        audio = batch
        if audio is None:
            continue
        process_file(audio[0].to('cpu'), processor, whisper_model, llama_model, accelerator, lora_config, optimizer, device, scaler)

    print("Processing complete!")

    # Save the fine-tuned LoRA weights
    save_fine_tuned_lora_weights(llama_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use for models")
    parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir, args.device, args.tsv_file)






















#use one gpu for whisper and another for llama
# '/data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips/common_voice_fr_20178526.mp3']
# 2024-07-11 19:16:40,852 - INFO - Processing audio and text
# 2024-07-11 19:16:40,853 - DEBUG - Audio tensor shape: torch.Size([480000])
# 2024-07-11 19:16:42,387 - DEBUG - STFT magnitudes shape: torch.Size([201, 3000])
# 2024-07-11 19:16:42,557 - DEBUG - Mel filters shape: torch.Size([80, 201])
# 2024-07-11 19:16:42,557 - DEBUG - Magnitudes shape before adjustment: torch.Size([201, 3000])
# 2024-07-11 19:16:42,557 - DEBUG - Target shape for adjustment: (201, 3000)
# 2024-07-11 19:16:43,052 - DEBUG - Mel spectrogram shape: torch.Size([80, 3000])
# 2024-07-11 19:16:47,532 - DEBUG - Embeddings shape: torch.Size([80, 1500, 1280])
# 2024-07-11 19:16:47,532 - ERROR - Failed to process audio: forward() missing 1 required positional argument: 'tgt'
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import argparse
# from functools import lru_cache
# from typing import Union, List
# import logging
# import ffmpeg
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast
# from accelerate import Accelerator
# from peft import get_peft_model, LoraConfig, TaskType
# import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# import pandas as pd
# import sys

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Hard-coded audio hyperparameters
# SAMPLE_RATE = 16000
# N_FFT = 400
# N_MELS = 80
# HOP_LENGTH = 160
# CHUNK_LENGTH = 30
# N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
# N_FRAMES = N_SAMPLES // HOP_LENGTH

# MAX_SEQ_LENGTH = 512  # Specify the maximum sequence length for tokenization

# class AudioDataset(Dataset):
#     def __init__(self, audio_files, text_data):
#         self.audio_files = audio_files
#         self.text_data = text_data

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         audio_file = self.audio_files[idx]
#         text = self.text_data[idx]
#         if not os.path.exists(audio_file):
#             logging.error(f"Audio file {audio_file} does not exist. Skipping this file.")
#             return None, None
#         audio = load_audio(audio_file)
#         audio = pad_or_trim(audio)
#         return audio, text

# def load_audio(file: str, sr: int = SAMPLE_RATE):
#     try:
#         out, _ = (
#             ffmpeg.input(file, threads=0)
#             .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
#             .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         logging.error(f"Failed to load audio {file}: {e.stderr.decode()}")
#         raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
#     if torch.is_tensor(array):
#         if array.shape[axis] > length:
#             array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
#         if array.shape[axis] < length:
#             pad_widths = [(0, 0)] * array.ndim
#             pad_widths[axis] = (0, length - array.shape[axis])
#             array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
#     else:
#         if array.shape[axis] > length:
#             array = array.take(indices=range(length), axis=axis)
#         if array.shape[axis] < length:
#             pad_widths = [(0, 0)] * array.ndim
#             pad_widths[axis] = (0, length - array.shape[axis])
#             array = np.pad(array, pad_widths)
#     return array

# @lru_cache(maxsize=None)
# def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
#     with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
#         return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

# def adjust_shape(input_tensor, target_shape, device):
#     input_tensor = input_tensor.cpu().numpy()
#     pad_widths = [(0, max(0, target - current)) for target, current in zip(target_shape, input_tensor.shape)]
#     adjusted_array = np.pad(input_tensor, pad_widths, mode='constant', constant_values=0)
#     adjusted_tensor = torch.tensor(adjusted_array).to(device)
#     return adjusted_tensor

# def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS, device='cuda'):
#     if not torch.is_tensor(audio):
#         if isinstance(audio, str):
#             audio = load_audio(audio)
#         audio = torch.from_numpy(audio)

#     if len(audio.shape) > 1:
#         audio = audio.squeeze()

#     audio = audio.to(device)  # Ensure audio tensor is moved to the CUDA device

#     logging.debug(f"Audio tensor shape: {audio.shape}")

#     window = torch.hann_window(N_FFT).to(device)
#     stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
#     magnitudes = stft[..., :-1].abs() ** 2

#     logging.debug(f"STFT magnitudes shape: {magnitudes.shape}")

#     filters = mel_filters(device, n_mels)

#     logging.debug(f"Mel filters shape: {filters.shape}")

#     magnitudes = magnitudes.to(device)

#     logging.debug(f"Magnitudes shape before adjustment: {magnitudes.shape}")
#     logging.debug(f"Target shape for adjustment: {(filters.shape[1], magnitudes.shape[1])}")

#     if magnitudes.shape[0] != filters.shape[1]:
#         magnitudes = adjust_shape(magnitudes, (filters.shape[1], magnitudes.shape[1]), device)
#         logging.debug(f"Magnitudes shape after adjustment: {magnitudes.shape}")

#     if filters.shape[1] != magnitudes.shape[0]:
#         raise ValueError(f"Shape mismatch: filters.shape[1]={filters.shape[1]}, magnitudes.shape[0]={magnitudes.shape[0]}")

#     mel_spec = filters @ magnitudes

#     logging.debug(f"Mel spectrogram shape: {mel_spec.shape}")

#     log_spec = torch.clamp(mel_spec, min=1e-10).log10()
#     log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
#     log_spec = (log_spec + 4.0) / 4.0
#     return log_spec

# def load_whisper_model(device='cuda:0'):
#     model_name = "openai/whisper-large-v3"
#     logging.debug(f"Loading Whisper model {model_name}")
#     try:
#         processor = WhisperProcessor.from_pretrained(model_name)
#         model = WhisperModel.from_pretrained(model_name).to(device)
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
#             torch.cuda.empty_cache()
#             try:
#                 processor = WhisperProcessor.from_pretrained(model_name)
#                 model = WhisperModel.from_pretrained(model_name).to(device)
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.error(f"CUDA out of memory again while loading Whisper model. Exiting.")
#                     sys.exit(1)
#                 else:
#                     logging.error(f"Failed to load Whisper model: {e}")
#                     sys.exit(1)
#         else:
#             logging.error(f"Failed to load Whisper model: {e}")
#             sys.exit(1)
#     logging.debug("Model loaded successfully")
#     return processor, model

# def get_audio_embeddings(audio, processor, model, device='cuda'):
#     mel_spec = log_mel_spectrogram(audio, device=device)
#     inputs = processor(mel_spec.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         embeddings = model.encoder(inputs['input_features']).last_hidden_state

#     logging.debug(f"Embeddings shape: {embeddings.shape}")
#     return embeddings

# class TLTRModel(torch.nn.Module):
#     # Placeholder TLTR Model implementation
#     def __init__(self):
#         super(TLTRModel, self).__init__()
#         self.transformer = torch.nn.Transformer()

#     def forward(self, x):
#         return self.transformer(x)

# class ProjectionLayer(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ProjectionLayer, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)

# def fine_tune_llama_with_lora(embeddings, text, llama_model, tokenizer, accelerator, lora_config, optimizer, device, scaler):
#     # Ensure the model has not been configured with multiple adapters
#     if not hasattr(llama_model, 'peft_config'):
#         llama_model = get_peft_model(llama_model, lora_config)

#     llama_model.train()

#     # Ensure that the tokenizer has a padding token
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     embeddings = embeddings.to(device)
#     with autocast():
#         outputs = llama_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
#         loss = outputs.loss

#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()

# def process_file(audio, text, processor, whisper_model, tltr_model, projection_layer, llama_model, tokenizer, accelerator, lora_config, optimizer, device, scaler):
#     logging.info("Processing audio and text")
#     try:
#         embeddings = get_audio_embeddings(audio, processor, whisper_model, device=device)
#         embeddings = tltr_model(embeddings)
#         embeddings = projection_layer(embeddings)
#         fine_tune_llama_with_lora(embeddings, text, llama_model, tokenizer, accelerator, lora_config, optimizer, device, scaler)
#         torch.cuda.empty_cache()
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logging.error("CUDA out of memory while processing audio. Attempting to clear cache and retry.")
#             torch.cuda.empty_cache()
#             try:
#                 embeddings = get_audio_embeddings(audio, processor, whisper_model, device=device)
#                 embeddings = tltr_model(embeddings)
#                 embeddings = projection_layer(embeddings)
#                 fine_tune_llama_with_lora(embeddings, text, llama_model, tokenizer, accelerator, lora_config, optimizer, device, scaler)
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.error("CUDA out of memory again while processing audio. Exiting.")
#                     sys.exit(1)
#                 else:
#                     logging.error(f"Failed to process audio: {e}")
#                     sys.exit(1)
#         else:
#             logging.error(f"Failed to process audio: {e}")
#             sys.exit(1)
#     except Exception as e:
#         logging.error(f"Failed to process audio: {e}")
#         sys.exit(1)

# def save_fine_tuned_lora_weights(llama_model, output_dir):
#     logging.info(f"Saving fine-tuned LoRA weights to {output_dir}")
#     lora_weights_path = os.path.join(output_dir, "lora_weights.bin")
#     torch.save(llama_model.state_dict(), lora_weights_path)
#     logging.info("LoRA weights saved successfully")

# def load_data_from_tsv(tsv_file: str, max_samples: int = 1000):
#     df = pd.read_csv(tsv_file, sep='\t')
#     df = df.sample(n=max_samples, random_state=42)
#     audio_files = df['path'].tolist()
#     texts = df['sentence'].tolist()
#     audio_files = [os.path.join(os.path.dirname(tsv_file), 'clips', path) for path in audio_files]
#     return audio_files, texts

# def main(audio_dir: str, output_dir: str, device: str, tsv_file: str):
#     logging.info(f"Starting processing on device {device}")

#     # Ensure the specified CUDA devices are valid
#     if not torch.cuda.is_available():
#         logging.error("CUDA is not available. Please check your CUDA installation.")
#         sys.exit(1)

#     try:
#         processor, whisper_model = load_whisper_model(device=device)
#     except Exception as e:
#         logging.error(f"Failed to load Whisper model: {e}")
#         sys.exit(1)

#     model_id = "meta-llama/Meta-Llama-3-8B"

#     try:
#         tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
#         llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
#         llama_model.to(device)
#     except Exception as e:
#         logging.error(f"Failed to load Llama model or tokenizer: {e}")
#         sys.exit(1)

#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias="none"
#     )

#     accelerator = Accelerator(device_placement=False)  # Disable automatic device placement by Accelerator
#     optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)
#     scaler = GradScaler()

#     # Prepare the model, optimizer, and possibly dataloaders
#     llama_model, optimizer = accelerator.prepare(llama_model, optimizer)

#     # Initialize TLTR and Projection Layer
#     tltr_model = TLTRModel().to(device)
#     projection_layer = ProjectionLayer(input_dim=1280, output_dim=llama_model.config.hidden_size).to(device)

#     audio_files, texts = load_data_from_tsv(tsv_file)

#     if not audio_files:
#         logging.error(f"No audio files found in the TSV file")
#         return

#     logging.info(f"Found {len(audio_files)} audio files")
#     logging.debug(f"Audio files: {audio_files}")

#     dataset = AudioDataset(audio_files, texts)
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

#     for batch in dataloader:
#         audio, text = batch
#         if audio is None or text is None:
#             continue
#         process_file(audio[0], text[0], processor, whisper_model, tltr_model, projection_layer, llama_model, tokenizer, accelerator, lora_config, optimizer, device, scaler)

#     print("Processing complete!")

#     # Save the fine-tuned LoRA weights
#     save_fine_tuned_lora_weights(llama_model, output_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
#     parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
#     parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
#     parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use for both Whisper and LLaMA models")
#     parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
#     args = parser.parse_args()

#     main(args.audio_dir, args.output_dir, args.device, args.tsv_file)









#use two gpus

# CUDA_VISIBLE_DEVICES=0,1 python3 /data/daisysxm76/speechtospeech/whisper/whisper/llama_whisper.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper --whisper_device cuda:0 --llama_device cuda:1 --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv

# CUDA_VISIBLE_DEVICES=5 python3 /data/daisysxm76/speechtospeech/whisper/whisper/llama_whisper.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper --device cuda:5 --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv

# CUDA_VISIBLE_DEVICES=1 python3 /data/daisysxm76/speechtospeech/whisper/whisper/llama_whisper.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv

# CUDA_VISIBLE_DEVICES=1 python3 /data/daisysxm76/speechtospeech/whisper/whisper/llama_whisper.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper  --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv
# source /data/daisysxm76/newenv/bin/activate
# CUDA_VISIBLE_DEVICES=2 python3 /data/daisysxm76/speechtospeech/whisper/whisper/llama_whisper.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper --device cuda:2 --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv

# before prejection

# import os
# import sys
# import argparse
# from functools import lru_cache
# from typing import List
# import logging
# import ffmpeg
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast
# from accelerate import Accelerator
# from peft import get_peft_model, LoraConfig, TaskType
# import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# import pandas as pd

# # Environment variable settings
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Hard-coded audio hyperparameters
# SAMPLE_RATE = 16000
# N_FFT = 400
# N_MELS = 80
# HOP_LENGTH = 160
# CHUNK_LENGTH = 30
# N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
# N_FRAMES = N_SAMPLES // HOP_LENGTH

# class AudioDataset(Dataset):
#     def __init__(self, audio_files):
#         self.audio_files = audio_files

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         return self.audio_files[idx]  # Return the file path


# def load_audio(file: str, sr: int = SAMPLE_RATE):
#     try:
#         out, _ = (
#             ffmpeg.input(file, threads=0)
#             .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
#             .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         logging.error(f"Failed to load audio {file}: {e.stderr.decode()}")
#         raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
#     if torch.is_tensor(array):
#         if array.shape[axis] > length:
#             array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
#         if array.shape[axis] < length:
#             pad_widths = [(0, 0)] * array.ndim
#             pad_widths[axis] = (0, length - array.shape[axis])
#             array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
#     else:
#         if array.shape[axis] > length:
#             array = array.take(indices=range(length), axis=axis)
#         if array.shape[axis] < length:
#             pad_widths = [(0, 0)] * array.ndim
#             pad_widths[axis] = (0, length - array.shape[axis])
#             array = np.pad(array, pad_widths)
#     return array

# @lru_cache(maxsize=None)
# def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
#     with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
#         return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

# def adjust_shape(input_tensor, target_shape, device):
#     input_tensor = input_tensor.cpu().numpy()
#     pad_widths = [(0, max(0, target - current)) for target, current in zip(target_shape, input_tensor.shape)]
#     adjusted_array = np.pad(input_tensor, pad_widths, mode='constant', constant_values=0)
#     adjusted_tensor = torch.tensor(adjusted_array).to(device)
#     return adjusted_tensor

# def load_whisper_model(device='cuda:0'):
#     model_name = "openai/whisper-large-v3"
#     logging.debug(f"Loading Whisper model {model_name}")
#     try:
#         processor = WhisperProcessor.from_pretrained(model_name)
#         model = WhisperModel.from_pretrained(model_name).to(device)
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
#             torch.cuda.empty_cache()
#             try:
#                 processor = WhisperProcessor.from_pretrained(model_name)
#                 model = WhisperModel.from_pretrained(model_name).to(device)
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.error(f"CUDA out of memory again while loading Whisper model. Exiting.")
#                     sys.exit(1)
#                 else:
#                     logging.error(f"Failed to load Whisper model: {e}")
#                     sys.exit(1)
#         else:
#             logging.error(f"Failed to load Whisper model: {e}")
#             sys.exit(1)
#     logging.debug("Model loaded successfully")
#     return processor, model

# def get_audio_embeddings(audio_file: str, processor, model, device='cuda'):
#     logging.debug(f"Processing audio file {audio_file}")
#     audio = load_audio(audio_file)
#     audio = pad_or_trim(audio)
#     audio = torch.from_numpy(audio).to(device)

#     inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         embeddings = model.encoder(inputs['input_features']).last_hidden_state

#     logging.debug(f"Embeddings shape: {embeddings.shape}")
#     return embeddings.cpu().numpy()  # Move embeddings to the CPU before returning

# def fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, device, scaler):
#     # Ensure the model has not been configured with multiple adapters
#     if not hasattr(llama_model, 'peft_config'):
#         llama_model = get_peft_model(llama_model, lora_config)

#     llama_model.train()

#     embeddings = embeddings.to(device)
#     with autocast():
#         outputs = llama_model(inputs_embeds=embeddings)
#         loss = outputs.loss

#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()

# def process_file(audio_file: str, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler):
#     logging.info(f"Processing audio file {audio_file}")
#     try:
#         embeddings = get_audio_embeddings(audio_file, processor, whisper_model, device=accelerator.device)
#         fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
#         torch.cuda.empty_cache()
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logging.error(f"CUDA out of memory while processing audio. Attempting to clear cache and retry.")
#             torch.cuda.empty_cache()
#             try:
#                 embeddings = get_audio_embeddings(audio_file, processor, whisper_model, device=accelerator.device)
#                 fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.error(f"CUDA out of memory again while processing audio. Exiting.")
#                     sys.exit(1)
#                 else:
#                     logging.error(f"Failed to process audio: {e}")
#                     sys.exit(1)
#         else:
#             logging.error(f"Failed to process audio: {e}")
#             sys.exit(1)
#     except Exception as e:
#         logging.error(f"Failed to process audio: {e}")
#         sys.exit(1)

# def save_fine_tuned_lora_weights(llama_model, output_dir):
#     logging.info(f"Saving fine-tuned LoRA weights to {output_dir}")
#     lora_weights_path = os.path.join(output_dir, "lora_weights.bin")
#     torch.save(llama_model.state_dict(), lora_weights_path)
#     logging.info("LoRA weights saved successfully")

# def load_data_from_tsv(tsv_file: str, audio_dir: str, max_samples: int = 1000):
#     df = pd.read_csv(tsv_file, sep='\t')

#     # Select the first max_samples entries
#     df = df.head(max_samples)

#     audio_files = df['path'].tolist()

#     # Filter out missing files
#     valid_audio_files = []
#     for audio_file in audio_files:
#         full_path = os.path.join(audio_dir, audio_file)
#         if os.path.exists(full_path):
#             valid_audio_files.append(full_path)
#         else:
#             logging.error(f"Audio file {full_path} does not exist.")
    
#     return valid_audio_files

# def main(audio_dir: str, output_dir: str, whisper_device: str, llama_device: str, tsv_file: str):
#     logging.info(f"Starting processing on devices whisper: {whisper_device}, llama: {llama_device}")

#     # Ensure the specified CUDA devices are valid
#     if not torch.cuda.is_available():
#         logging.error("CUDA is not available. Please check your CUDA installation.")
#         sys.exit(1)

#     available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
#     logging.info(f"Available CUDA devices: {available_devices}")

#     try:
#         processor, whisper_model = load_whisper_model(device=whisper_device)
#     except Exception as e:
#         logging.error(f"Failed to load Whisper model: {e}")
#         sys.exit(1)

#     model_id = "meta-llama/Meta-Llama-3-8B"

#     try:
#         llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
#         llama_model.to(llama_device)
#     except Exception as e:
#         logging.error(f"Failed to load Llama model: {e}")
#         sys.exit(1)

#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias="none"
#     )

#     accelerator = Accelerator(device_placement=False)  # Disable automatic device placement by Accelerator
#     optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)
#     scaler = GradScaler()

#     # Prepare the model, optimizer, and possibly dataloaders
#     llama_model, optimizer = accelerator.prepare(llama_model, optimizer)

#     audio_files = load_data_from_tsv(tsv_file, audio_dir)

#     if not audio_files:
#         logging.error(f"No audio files found in the TSV file")
#         return

#     logging.info(f"Found {len(audio_files)} valid audio files")
#     logging.debug(f"Valid audio files: {audio_files}")

#     dataset = AudioDataset(audio_files)
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

#     for batch in dataloader:
#         audio_file = batch[0]  # Get the file path
#         process_file(audio_file, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)

#     print("Processing complete!")

#     # Save the fine-tuned LoRA weights
#     save_fine_tuned_lora_weights(llama_model, output_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
#     parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
#     parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
#     parser.add_argument("--whisper_device", type=str, default="cuda:6", help="CUDA device to use for Whisper model")
#     parser.add_argument("--llama_device", type=str, default="cuda:7", help="CUDA device to use for LLaMA model")
#     parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
#     args = parser.parse_args()

#     main(args.audio_dir, args.output_dir, args.whisper_device, args.llama_device, args.tsv_file)

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
from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast
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

    return torch.frombuffer(out, dtype=torch.int16).float().div(32768.0)  # Convert to float32, normalize

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if array.shape[axis] > length:
        array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
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
    adjusted_tensor = F.pad(input_tensor, pad_widths, mode='constant', value=0).to(device)
    return adjusted_tensor

def load_whisper_model(device='cuda:0'):
    model_name = "openai/whisper-large-v3"
    logging.debug(f"Loading Whisper model {model_name}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperModel.from_pretrained(model_name).to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA out of memory again while loading Whisper model. Exiting.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to load Whisper model: {e}")
                    sys.exit(1)
        else:
            logging.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    logging.debug("Model loaded successfully")
    return processor, model

def get_audio_embeddings(audio_file: str, processor, model, device='cuda'):
    logging.debug(f"Processing audio file {audio_file}")
    audio = load_audio(audio_file)
    audio = pad_or_trim(audio).to(device)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model.encoder(inputs['input_features']).last_hidden_state

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

def process_file(audio_file: str, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler, projection_matrix):
    logging.info(f"Processing audio file {audio_file}")
    try:
        embeddings = get_audio_embeddings(audio_file, processor, whisper_model, device=accelerator.device)
        embeddings = project_embeddings(embeddings, projection_matrix).to(llama_device)
        fine_tune_llama_with_lora(embeddings, llama_model, accelerator, lora_config, optimizer, llama_device, scaler)
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while processing audio. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                embeddings = get_audio_embeddings(audio_file, processor, whisper_model, device=accelerator.device)
                embeddings = project_embeddings(embeddings, projection_matrix).to(llama_device)
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
        processor, whisper_model = load_whisper_model(device=whisper_device)
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
        process_file(audio_file, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler, projection_matrix)

    print("Processing complete!")

    # Save the fine-tuned LoRA weights
    save_fine_tuned_lora_weights(llama_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
    parser.add_argument("--whisper_device", type=str, default="cuda:0", help="CUDA device to use for Whisper model")
    parser.add_argument("--llama_device", type=str, default="cuda:5", help="CUDA device to use for LLaMA model")
    parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir, args.whisper_device, args.llama_device, args.tsv_file)




"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from functools import lru_cache
from typing import Union
import logging
import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast, pipeline
import bitsandbytes as bnb
from datasets import load_dataset
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import sys
import torch.optim as optim

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

def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS, device='cuda'):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio).to(device)

    if len(audio.shape) > 1:
        audio = audio.squeeze()

    logging.debug(f"Audio tensor shape: {audio.shape}")

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

def load_whisper_model(device='cuda:0'):
    model_name = "openai/whisper-large-v3"
    logging.debug(f"Loading Whisper model {model_name}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperModel.from_pretrained(model_name).to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA out of memory again while loading Whisper model. Exiting.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to load Whisper model: {e}")
                    sys.exit(1)
        else:
            logging.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    logging.debug("Model loaded successfully")
    return processor, model

def get_audio_embeddings(audio_file: str, processor, model, device='cuda'):
    logging.debug(f"Processing audio file {audio_file}")
    audio = load_audio(audio_file)
    audio = pad_or_trim(audio)
    mel_spec = log_mel_spectrogram(audio, device=device)
    inputs = processor(mel_spec.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model.encoder(inputs['input_features']).last_hidden_state
    
    logging.debug(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def fine_tune_llama_with_lora(embeddings, llama_model, tokenizer, accelerator, lora_config, optimizer, device):
    # Ensure the model has not been configured with multiple adapters
    if not hasattr(llama_model, 'peft_config'):
        llama_model = get_peft_model(llama_model, lora_config)
    
    llama_model.train()

    # Example text data to use for fine-tuning
    text_data = ["Example text for fine-tuning LLaMA model."]
    
    # Ensure that the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    inputs = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Combine embeddings with text inputs for fine-tuning
    embeddings = embeddings.to(device)
    outputs = llama_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])

    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def process_file(file, processor, whisper_model, llama_model, tokenizer, accelerator, lora_config, optimizer, llama_device):
    logging.info(f"Processing file {file}")
    try:
        embeddings = get_audio_embeddings(file, processor, whisper_model, device=accelerator.device)
        fine_tune_llama_with_lora(embeddings, llama_model, tokenizer, accelerator, lora_config, optimizer, llama_device)
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while processing {file}. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            # Retry once
            try:
                embeddings = get_audio_embeddings(file, processor, whisper_model, device=accelerator.device)
                fine_tune_llama_with_lora(embeddings, llama_model, tokenizer, accelerator, lora_config, optimizer, llama_device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA out of memory again while processing {file}. Exiting.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to process {file}: {e}")
                    sys.exit(1)
        else:
            logging.error(f"Failed to process {file}: {e}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to process {file}: {e}")
        sys.exit(1)

def save_fine_tuned_lora_weights(llama_model, output_dir):
    logging.info(f"Saving fine-tuned LoRA weights to {output_dir}")
    lora_weights_path = os.path.join(output_dir, "lora_weights.bin")
    torch.save(llama_model.state_dict(), lora_weights_path)
    logging.info("LoRA weights saved successfully")

def main(audio_dir: str, output_dir: str, whisper_device: str, llama_device: str):
    logging.info(f"Starting processing on devices whisper: {whisper_device}, llama: {llama_device}")

    # Ensure the specified CUDA devices are valid
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Please check your CUDA installation.")
        sys.exit(1)

    available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available CUDA devices: {available_devices}")

    try:
        processor, whisper_model = load_whisper_model(device=whisper_device)
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)

    model_id = "meta-llama/Meta-Llama-3-8B"

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        llama_model.to(llama_device)
    except Exception as e:
        logging.error(f"Failed to load Llama model or tokenizer: {e}")
        sys.exit(1)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )

    accelerator = Accelerator(device_placement=False)  # Disable automatic device placement by Accelerator
    optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)

    # Prepare the model, optimizer, and possibly dataloaders
    llama_model, optimizer = accelerator.prepare(llama_model, optimizer)

    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]

    if not audio_files:
        logging.error(f"No audio files found in directory {audio_dir}")
        return

    logging.info(f"Found {len(audio_files)} audio files")
    logging.debug(f"Audio files: {audio_files}")

    total_files = len(audio_files)  # Total number of files to process
    counter = 0  # Initialize counter

    for file in audio_files:
        process_file(file, processor, whisper_model, llama_model, tokenizer, accelerator, lora_config, optimizer, llama_device)
        counter += 1
        percentage = (counter / total_files) * 100
        print(f"Processing: {percentage:.2f}%", end='\r', flush=True)

    print("Processing complete!")

    # Save the fine-tuned LoRA weights
    save_fine_tuned_lora_weights(llama_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
    parser.add_argument("--whisper_device", type=str, default="cuda:6", help="CUDA device to use for Whisper model")
    parser.add_argument("--llama_device", type=str, default="cuda:7", help="CUDA device to use for LLaMA model")
    args = parser.parse_args()

    main(args.audio_dir, args.output_dir, args.whisper_device, args.llama_device)


"""