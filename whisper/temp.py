# CUDA_VISIBLE_DEVICES=1 python3 /data/daisysxm76/speechtospeech/whisper/whisper/temp.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/fine_tuned_model_whisper --tsv_file /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train_first1000_translate.tsv
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from functools import lru_cache
import logging
import ffmpeg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, WhisperModel, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
#from tltr import TLTRModel  # Assuming TLTRModel is defined in a module named tltr

from config import SongConfig

# Initialize the config
config = SongConfig()

from audio import AudioEncoder
AudioEncoder = AudioEncoder(**config.audio) # 5 required positional arguments: 'n_mels', 'n_ctx', 'n_state', 'n_head', and 'n_layer'
#AudioEncoder = AudioEncoder()

from models import TLTR
tltr_model = TLTR()  # Initialize TLTR model

# Initialize the GradScaler torch.cuda.amp with an optimizer that hasn't been properly set up for mixed precision training. 
# This involves ensuring that the GradScaler is correctly interacting with the optimizer.
scaler = torch.cuda.amp.GradScaler()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging configuration is loaded
logging.debug("Logging is configured and debug mode is enabled")

# Hard-coded audio hyperparameters
SAMPLE_RATE = 16000 # 16kHz sample rate
N_FFT = 400 # 25ms window/ window size for each Fast Fourier Transform
N_MELS = 80 #The mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another
HOP_LENGTH = 160    #the number of samples between successive frames
CHUNK_LENGTH = 30   #The length of each audio chunk to be processed

def exact_div(x, y):
    assert x % y == 0
    return x // y

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input 
#N_FRAMES = N_SAMPLES // HOP_LENGTH

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

class AudioDataset(Dataset):
    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):  # Return audio tensor
        audio_file = self.audio_files[idx]
        if not os.path.exists(audio_file):
            return None
        audio = load_audio(audio_file)  # Load audio file as np.ndarray
        return audio  # Return the loaded audio as np.ndarray

def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file using ffmpeg, convert to mono and resample.
    """
    logging.debug(f"Loading audio file: {file_path}")
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
    logging.debug(f"Loading audio: {audio}")
    return audio
        
@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    filters = torch.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.pt"))
    return filters[f"mel_{n_mels}"].to(device)

def load_whisper_model(device='cuda:0'):
    model_name = "openai/whisper-large-v3"
    logging.debug(f"Loading Whisper model {model_name}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device).to(torch.float16)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"CUDA out of memory while loading Whisper model. Attempting to clear cache and retry.")
            torch.cuda.empty_cache()
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperModel.from_pretrained(model_name).to(device).to(torch.float16)
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

def get_audio_embeddings(audio_file: str, processor, model) -> torch.Tensor:
    logging.debug(f"Loading audio file: {audio_file}")
    audio = load_audio(audio_file)
    logging.debug(f"Audio loaded: {audio.shape}, dtype: {audio.dtype}")
   
    # Convert the NumPy array to a PyTorch tensor if needed.
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    logging.debug(f"Audio tensor shape: {audio.shape}, dtype: {audio.dtype}")

    # Ensure the tensor is on the CPU before processing
    if audio.device.type == 'cuda':
        audio = audio.cpu()

    # Process the audio tensor with the processor
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to("cuda").to(torch.float16)
    logging.debug(f"Processor inputs prepared for: {audio_file}, inputs shape: {inputs.shape}, dtype: {inputs.dtype}")

    with torch.no_grad():
        embeddings = model.encoder(inputs).last_hidden_state.to(torch.float16)
        logging.debug(f"Embeddings generated: {embeddings.shape}")
    # embeddings = AudioEncoder.encode(audio)
    logging.debug(f"Embeddings generated for $$$$: {audio_file}, embeddings shape: {embeddings.shape}")
    
    if embeddings is None:
        logging.error(f"Skipping audio file {audio} due to failed embeddings generation.")
        return
    
     # Pass the embeddings through TLTR model
    logging.debug(f"Passing embeddings through TLTR model")
    tltr_outputs = tltr_model.to('cuda')(embeddings.to('cuda'))
    logging.debug(f"TLTR embeddings generated: {embeddings}")

    # Combine Whisper and TLTR outputs
    embeddings = torch.cat((embeddings, tltr_outputs), dim=-1)
    return embeddings

def save_fine_tuned_lora_weights(llama_model, output_dir):
    logging.info(f"Saving fine-tuned LoRA weights to {output_dir}")
    lora_weights_path = os.path.join(output_dir, "lora_weights.bin")
    torch.save(llama_model.state_dict(), lora_weights_path)
    logging.info("LoRA weights saved successfully")

def load_data_from_tsv(tsv_file: str, audio_dir: str, max_samples: int = 100):
    df = pd.read_csv(tsv_file, sep='\t')
    df = df.head(max_samples)
    audio_files = [os.path.join(audio_dir, file) for file in df['path'].tolist()]
    return audio_files

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def create_dataset(df, tokenizer):
    tokenized_data = tokenize_sentences(df["translation split"].tolist(), tokenizer)
    dataset = Dataset.from_dict({
        "path": df["path"].tolist(),
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"]
    })
    dataset_dict = DatasetDict({
        "train": dataset
    })
    return dataset_dict

def fine_tune_llama_with_lora(embeddings, llama_model, lora_config, optimizer, device, scaler, target_translations, tokenizer):
    # Apply LoRA configuration if not already present
    if not hasattr(llama_model, 'peft_config'):
        logging.debug("Applying LoRA configuration")
        llama_model = get_peft_model(llama_model, lora_config)

    logging.debug(f"Starting fine-tune with LoRA, embeddings shape: {embeddings.shape}")

    # Move embeddings to the specified device and ensure dtype is float16
    embeddings = embeddings.to(device).to(torch.float16)
    logging.debug(f"Embeddings moved to device: {embeddings.device}")

    # Ensure embeddings have the correct shape
    if embeddings.size(-1) != 4096:
        projection_layer = torch.nn.Linear(embeddings.size(-1), 4096).to(device).to(torch.float16)
        embeddings = projection_layer(embeddings)
    logging.debug(f"Projected embeddings shape: {embeddings.shape}")

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the target translations
    tokenized_output = tokenizer(target_translations, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokenized_output['input_ids'].to(device)
    attention_mask = tokenized_output['attention_mask'].to(device)
    logging.debug(f"Tokenized target translations: {input_ids.shape}")

    # Align input_ids sequence length with embeddings sequence length
    if input_ids.size(1) > embeddings.size(1):
        logging.debug(f"Trimmed input_ids shape: {input_ids.shape}")
        input_ids = input_ids[:, :embeddings.size(1)]
        
        exit()
    elif input_ids.size(1) < embeddings.size(1):
        logging.debug(f"embeddings.shape: {embeddings.shape}")
        logging.debug(f"Padded input_ids shape: {input_ids.shape}")
        padding_size = embeddings.size(1) - input_ids.size(1)
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_size), value=tokenizer.pad_token_id)
        exit()
    logging.debug(f"Aligned input_ids shape: {input_ids.shape}")

    llama_model.train()

    try:
        # Removing autocast context
        outputs = llama_model(inputs_embeds=embeddings, labels=input_ids)
        loss = outputs.loss if hasattr(outputs, 'loss') else None

        if loss is None:
            loss_fn = torch.nn.CrossEntropyLoss()
            logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
            labels_flat = input_ids.view(-1)
            loss = loss_fn(logits_flat, labels_flat)
            logging.debug(f"Computed loss manually: {loss}")

        if loss is None:
            raise ValueError("Loss is None. Ensure that the model is computing the loss correctly.")
        logging.debug(f"loss: {loss}")

        # Directly backward without scaling for debugging
        loss.backward()
        logging.debug("Performed backward pass")

        optimizer.step()
        logging.debug("Optimizer step completed")

        optimizer.zero_grad()
        logging.debug("Gradients zeroed out")
    except Exception as e:
        logging.error(f"An error occurred during the optimization process: {e}")
        raise

    return llama_model


def process_file(audio, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, llama_device, scaler, tokenizer):
    logging.info(f"Processing audio: {audio}")
    # find the target translation for the audio
    file_path = "/data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train_first1000_translate.tsv"
    # Load the TSV file into a DataFrame
    data = pd.read_csv(file_path, sep='\t')
    # Extract the 'audio file name' and 'translation' columns
    file_name = os.path.basename(audio)
    # Find the translation for the given audio file
    translation_row = data[data['path'] == file_name]
    # Log the contents of translation_row
    logging.debug(f"Translation row for {file_name}: \n{translation_row}")
    if not translation_row.empty:
        audio_translations = translation_row.iloc[0]['translation']
        logging.debug(f"Audio translations: {audio} {audio_translations}")
    
        try:
            embeddings = get_audio_embeddings(audio, processor, whisper_model)
            logging.debug(f"Embeddings generated for####, embeddings shape: {embeddings.shape}")

            logging.debug(f"starting fine tune llama with lora, embeddings shape: {embeddings.shape}")
            fine_tune_llama_with_lora(embeddings, llama_model, lora_config, optimizer, llama_device, scaler, audio_translations, tokenizer)
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error(f"CUDA out of memory while processing audio. Attempting to clear cache and retry.")
                torch.cuda.empty_cache()
                try:
                    embeddings = get_audio_embeddings(audio, processor, whisper_model)
                    logging.debug(f"Embeddings generated for: {audio}, embeddings shape: {embeddings.shape}")

                    fine_tune_llama_with_lora(embeddings, llama_model, lora_config, optimizer, llama_device, scaler, audio_translations, tokenizer)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error(f"CUDA out of memory again while processing audio. Exiting.")
                        sys.exit(1)
                    else:
                        logging.error(f"Failed to process audio 1: {e}")
                        sys.exit(1)
            else:
                logging.error(f"Failed to process audio 2: {e}")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to process audio 3: {e}")
            logging.error(f"Type of audio: {type(audio)}")
            sys.exit(1)
    else:
        logging.error(f"No translation found for audio: {audio}")

def main(audio_dir: str, output_dir: str, device: str, tsv_file: str):
    logging.info(f"Starting processing on device {device}")
    scaler = GradScaler()

    available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available CUDA devices: {available_devices}")

    try:
        processor, whisper_model = load_whisper_model(device=device)
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)

    model_id = "meta-llama/Meta-Llama-3-8B"

    try:
        llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        llama_model.to(device)
    except Exception as e:
        logging.error(f"Failed to load Llama model: {e}")
        sys.exit(1)

    freeze_model(llama_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )

    accelerator = Accelerator(device_placement=False)
    optimizer = optim.AdamW(llama_model.parameters(), lr=1e-4)

    # Prepare model and optimizer with accelerator
    llama_model, optimizer = accelerator.prepare(llama_model, optimizer)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    audio_files = load_data_from_tsv(tsv_file, audio_dir)

    if not audio_files:
        logging.error(f"No audio files found in the TSV file")
        return

    logging.info(f"Found {len(audio_files)} audio files")
    logging.debug(f"Audio files: {audio_files}")

    for file in audio_files:
        if file is None:
            continue
        logging.debug(f"file: {file}")
        process_file(file, processor, whisper_model, llama_model, accelerator, lora_config, optimizer, device, scaler, tokenizer)

    print("Processing complete!")

    save_fine_tuned_lora_weights(llama_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and fine-tune LLama model with LoRA using audio embeddings.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the fine-tuned LoRA weights")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use for models")
    parser.add_argument("--tsv_file", type=str, help="Path to the TSV file containing the dataset")
    args = parser.parse_args()
    
    main(args.audio_dir, args.output_dir, args.device, args.tsv_file)


