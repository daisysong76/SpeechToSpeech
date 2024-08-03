# Install necessary packages
"""pip install datasets
pip install accelerate
pip install hydra-core
pip install higher
pip install bitsandbytes
pip install peft
pip install torchaudio

"""
# CUDA_VISIBLE_DEVICES=7 python3 /data/daisysxm76/speechtospeech/whisper/whisper/hubert_llama.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/embeddings --num_workers 4
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, HubertModel, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torchaudio
from torch.cuda.amp import GradScaler, autocast
import pickle
from peft import get_peft_model, LoraConfig

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the process group for Distributed Data Parallel
dist.init_process_group(backend='nccl')

# Function to load Hubert model
def load_hubert_model(model_name: str = "facebook/hubert-base-ls960"):
    model = HubertModel.from_pretrained(model_name)
    model.to(device)
    return model

# Load Hubert model
hubert_model = load_hubert_model()

# Feature extractor for Hubert
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

def extract_and_cache_features(file_path, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(file_path) + '.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
    else:
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=sample_rate).to(device)
        with torch.no_grad():
            features = hubert_model(**inputs).last_hidden_state
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    return features

# Dataset class
class MP3Dataset(Dataset):
    def __init__(self, file_paths, feature_extractor, hubert_model, cache_dir='cache'):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.hubert_model = hubert_model
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = extract_and_cache_features(file_path, self.cache_dir)
        return features

def main(mp3_directory, embeddings_dir, num_workers):
    try:
        # Get all MP3 file paths
        mp3_file_paths = [os.path.join(mp3_directory, f) for f in os.listdir(mp3_directory) if f.endswith('.mp3')]

        # Create dataset and dataloader
        mp3_dataset = MP3Dataset(mp3_file_paths, feature_extractor, hubert_model)
        data_loader = DataLoader(mp3_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

        # TLTR Model
        class TLTRModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(TLTRModel, self).__init__()
                self.transformer = nn.Transformer(d_model=input_dim, nhead=8, num_encoder_layers=6)
                self.projection = nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                transformer_output = self.transformer(x)
                projected_output = self.projection(transformer_output)
                return projected_output

        # Initialize TLTR model
        input_dim = 768  # Assuming Hubert model output dimension
        output_dim = 4096  # Matching LLaMA embedding dimension
        tltr_model = TLTRModel(input_dim, output_dim).to(device)

        # Concatenate Audio and Text Tokens
        def prepare_input(audio_features, text_tokens):
            audio_tokens = tltr_model(audio_features.to(device))
            input_tokens = torch.cat([audio_tokens, text_tokens.to(device)], dim=1)
            return input_tokens

        # Assuming text data has been tokenized using LLaMA tokenizer
        # Example tokenized text data
        tokenized_text_data = [torch.tensor([[1, 2, 3]]).to(device), torch.tensor([[4, 5, 6]]).to(device)]

        # Example usage
        input_tokens = prepare_input(audio_features[0], tokenized_text_data[0])

        # Load LLaMA model for causal language modeling
        llama_model = AutoModelForCausalLM.from_pretrained("facebook/llama-3-8b").to(device)

        # Apply LoRA adapters using PEFT (Parameter-Efficient Fine-Tuning)
        peft_config = LoraConfig(
            r=16,  # Rank of the low-rank adaptation matrices
            lora_alpha=32,
            lora_dropout=0.1,
        )

        llama_model = get_peft_model(llama_model, peft_config)

        # Apply LoRA adapters (Low-Rank Adaptation)
        for name, param in llama_model.named_parameters():
            if 'attention' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir='./logs',
            logging_steps=10,
            gradient_accumulation_steps=16,
            fp16=True,
        )

        # Assuming the datasets are already tokenized and available as train and eval datasets
        # Placeholder train and eval datasets
        train_dataset = torch.utils.data.TensorDataset(input_tokens)
        eval_dataset = torch.utils.data.TensorDataset(input_tokens)

        # Initialize Trainer
        trainer = Trainer(
            model=llama_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model using validation set
        # Placeholder validation and unseen datasets
        validation_dataset = eval_dataset
        unseen_dataset = eval_dataset

        # Function to evaluate the model
        def evaluate_model(trainer, validation_dataset):
            trainer.evaluate(validation_dataset)

        # Function to test the model on unseen data
        def test_model(trainer, unseen_dataset):
            trainer.predict(unseen_dataset)

        # Evaluate and test the model
        evaluate_model(trainer, validation_dataset)
        test_model(trainer, unseen_dataset)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_directory", type=str, help="Directory containing MP3 files")
    parser.add_argument("embeddings_dir", type=str, help="Directory to save embeddings")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()

    main(args.mp3_directory, args.embeddings_dir, args.num_workers)
"""
#CUDA_VISIBLE_DEVICES=6,7 python3 /data/daisysxm76/speechtospeech/whisper/whisper/hubert_llama.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/embeddings --num_workers 4
# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 /data/daisysxm76/speechtospeech/whisper/whisper/hubert_llama.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips /data/daisysxm76/speechtospeech/dataset_fr_en/embeddings --num_workers 4
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import Wav2Vec2FeatureExtractor, HubertModel, AutoModelForCausalLM, Trainer, TrainingArguments
import torchaudio
from torch.cuda.amp import GradScaler, autocast
import pickle
from peft import get_peft_model, LoraConfig

def load_hubert_model(model_name: str = "facebook/hubert-base-ls960"):
    model = HubertModel.from_pretrained(model_name)
    return model

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# Resample to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

def extract_and_cache_features(file_path, hubert_model, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(file_path) + '.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
    else:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = resampler(waveform)
        inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            features = hubert_model(**inputs).last_hidden_state
            print(f"Extracted features shape: {features.shape}")  # Debugging print statement
            # Ensure the shape is [batch_size, sequence_length, feature_dim]
            if features.dim() == 4:
                features = features.squeeze(1)
            print(f"Features shape after squeeze: {features.shape}")  # Debugging print statement
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    return features

class MP3Dataset(Dataset):
    def __init__(self, file_paths, feature_extractor, hubert_model, cache_dir='cache'):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.hubert_model = hubert_model
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = extract_and_cache_features(file_path, self.hubert_model, self.cache_dir)
        return features

def main(mp3_directory, embeddings_dir, num_workers, local_rank):
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        hubert_model = load_hubert_model().to(device)

        mp3_file_paths = [os.path.join(mp3_directory, f) for f in os.listdir(mp3_directory) if f.endswith('.mp3')]

        mp3_dataset = MP3Dataset(mp3_file_paths, feature_extractor, hubert_model)
        sampler = DistributedSampler(mp3_dataset, num_replicas=world_size, rank=rank)
        data_loader = DataLoader(mp3_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)

        class TLTRModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(TLTRModel, self).__init__()
                self.transformer = nn.Transformer(d_model=input_dim, nhead=8, num_encoder_layers=6)
                self.projection = nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                print(f"TLTRModel input shape: {x.shape}")  # Debugging print statement
                transformer_output = self.transformer(x)
                projected_output = self.projection(transformer_output)
                return projected_output

        input_dim = 768
        output_dim = 4096
        tltr_model = TLTRModel(input_dim, output_dim).to(device)
        tltr_model = DDP(tltr_model, device_ids=[local_rank])

        def prepare_input(audio_features, text_tokens):
            # Ensure audio_features has the shape [batch_size, sequence_length, feature_dim]
            if audio_features.dim() == 4:
                audio_features = audio_features.squeeze(1)
            if audio_features.dim() == 3:
                audio_features = audio_features.permute(0, 2, 1)  # [batch_size, sequence_length, feature_dim] to [batch_size, feature_dim, sequence_length]
            print(f"Audio features shape before TLTRModel: {audio_features.shape}")  # Debugging print statement
            audio_tokens = tltr_model(audio_features.to(device))
            print(f"Audio tokens shape after TLTRModel: {audio_tokens.shape}")  # Debugging print statement
            input_tokens = torch.cat([audio_tokens, text_tokens.to(device)], dim=1)
            return input_tokens

        tokenized_text_data = [torch.tensor([[1, 2, 3]]).to(device), torch.tensor([[4, 5, 6]]).to(device)]

        audio_features = extract_and_cache_features(mp3_file_paths[0], hubert_model, cache_dir='cache')

        input_tokens = prepare_input(audio_features, tokenized_text_data[0])

        llama_model = AutoModelForCausalLM.from_pretrained("/data/akshat/models/Meta_Llama-3-8B").to(device)
        llama_model = DDP(llama_model, device_ids=[local_rank])

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        llama_model = get_peft_model(llama_model, peft_config)

        for name, param in llama_model.named_parameters():
            if 'attention' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir='./logs',
            logging_steps=10,
            gradient_accumulation_steps=16,
            fp16=True,
            dataloader_drop_last=True,
        )

        train_dataset = torch.utils.data.TensorDataset(input_tokens)
        eval_dataset = torch.utils.data.TensorDataset(input_tokens)

        trainer = Trainer(
            model=llama_model.module,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        validation_dataset = eval_dataset
        unseen_dataset = eval_dataset

        def evaluate_model(trainer, validation_dataset):
            trainer.evaluate(validation_dataset)

        def test_model(trainer, unseen_dataset):
            trainer.predict(unseen_dataset)

        evaluate_model(trainer, validation_dataset)
        test_model(trainer, unseen_dataset)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_directory", type=str, help="Directory containing MP3 files")
    parser.add_argument("embeddings_dir", type=str, help="Directory to save embeddings")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--local_rank", type=int, default=local_rank, help="Local rank for distributed training")
    args = parser.parse_args()

    main(args.mp3_directory, args.embeddings_dir, args.num_workers, args.local_rank)




"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import Wav2Vec2FeatureExtractor, HubertModel, AutoModelForCausalLM, Trainer, TrainingArguments
import torchaudio
from torch.cuda.amp import GradScaler, autocast
import pickle
from peft import get_peft_model, LoraConfig

def load_hubert_model(model_name: str = "facebook/hubert-base-ls960"):
    model = HubertModel.from_pretrained(model_name)
    return model

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# Resample to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

def extract_and_cache_features(file_path, hubert_model, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(file_path) + '.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
    else:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = resampler(waveform)
        inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            features = hubert_model(**inputs).last_hidden_state
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    return features

class MP3Dataset(Dataset):
    def __init__(self, file_paths, feature_extractor, hubert_model, cache_dir='cache'):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.hubert_model = hubert_model
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = extract_and_cache_features(file_path, self.hubert_model, self.cache_dir)
        return features

def main(mp3_directory, embeddings_dir, num_workers, local_rank):
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        hubert_model = load_hubert_model().to(device)

        mp3_file_paths = [os.path.join(mp3_directory, f) for f in os.listdir(mp3_directory) if f.endswith('.mp3')]

        mp3_dataset = MP3Dataset(mp3_file_paths, feature_extractor, hubert_model)
        sampler = DistributedSampler(mp3_dataset, num_replicas=world_size, rank=rank)
        data_loader = DataLoader(mp3_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)

        class TLTRModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(TLTRModel, self).__init__()
                self.transformer = nn.Transformer(d_model=input_dim, nhead=8, num_encoder_layers=6)
                self.projection = nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                transformer_output = self.transformer(x)
                projected_output = self.projection(transformer_output)
                return projected_output

        input_dim = 768
        output_dim = 4096
        tltr_model = TLTRModel(input_dim, output_dim).to(device)
        tltr_model = DDP(tltr_model, device_ids=[local_rank])

        def prepare_input(audio_features, text_tokens):
            audio_tokens = tltr_model(audio_features.to(device))
            input_tokens = torch.cat([audio_tokens, text_tokens.to(device)], dim=1)
            return input_tokens

        tokenized_text_data = [torch.tensor([[1, 2, 3]]).to(device), torch.tensor([[4, 5, 6]]).to(device)]

        audio_features = extract_and_cache_features(mp3_file_paths[0], hubert_model, cache_dir='cache')

        input_tokens = prepare_input(audio_features, tokenized_text_data[0])

        llama_model = AutoModelForCausalLM.from_pretrained("/data/akshat/models/Meta_Llama-3-8B").to(device)
        llama_model = DDP(llama_model, device_ids=[local_rank])

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        llama_model = get_peft_model(llama_model, peft_config)

        for name, param in llama_model.named_parameters():
            if 'attention' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir='./logs',
            logging_steps=10,
            gradient_accumulation_steps=16,
            fp16=True,
            dataloader_drop_last=True,
        )

        train_dataset = torch.utils.data.TensorDataset(input_tokens)
        eval_dataset = torch.utils.data.TensorDataset(input_tokens)

        trainer = Trainer(
            model=llama_model.module,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        validation_dataset = eval_dataset
        unseen_dataset = eval_dataset

        def evaluate_model(trainer, validation_dataset):
            trainer.evaluate(validation_dataset)

        def test_model(trainer, unseen_dataset):
            trainer.predict(unseen_dataset)

        evaluate_model(trainer, validation_dataset)
        test_model(trainer, unseen_dataset)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_directory", type=str, help="Directory containing MP3 files")
    parser.add_argument("embeddings_dir", type=str, help="Directory to save embeddings")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--local_rank", type=int, default=local_rank, help="Local rank for distributed training")
    args = parser.parse_args()

    main(args.mp3_directory, args.embeddings_dir, args.num_workers, args.local_rank)


"""
