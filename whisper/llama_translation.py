"""
import os
import argparse
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def load_llama_model(model_name: str = "huggingface/llama-3b"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def translate_embeddings_to_text(embeddings, tokenizer, model):
    inputs = tokenizer(embeddings, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main(embeddings_file: str):
    embeddings = torch.load(embeddings_file)
    
    llama_tokenizer, llama_model = load_llama_model()
    translated_text = translate_embeddings_to_text(embeddings, llama_tokenizer, llama_model)
    print(translated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate audio embeddings to text using LLaMA.")
    parser.add_argument("embeddings_file", type=str, help="Path to the saved embeddings file")
    args = parser.parse_args()

    main(args.embeddings_file)

"""

# batching process
import os
import argparse
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import multiprocessing as mp
from tqdm import tqdm

def load_llama_model(model_name: str = "huggingface/llama-3b"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")
    return tokenizer, model

def translate_embeddings_to_text(embeddings, tokenizer, model):
    inputs = tokenizer(embeddings, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def process_file(file, tokenizer, model, output_dir, lock, counter):
    try:
        embeddings = torch.load(file)
        translated_text = translate_embeddings_to_text(embeddings, tokenizer, model)
        output_file = os.path.join(output_dir, os.path.basename(file) + ".txt")
        with open(output_file, "w") as f:
            f.write(translated_text)
        print(f"Translation saved to {output_file}")

        with lock:
            counter.value += 1
            if counter.value % 10 == 0:
                print(f"Processed {counter.value} files, saving intermediate results...")

    except Exception as e:
        print(f"Failed to process {file}: {e}")

def main(embeddings_dir: str, output_dir: str, num_workers: int):
    llama_tokenizer, llama_model = load_llama_model()
    embedding_files = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith(".pt")]

    os.makedirs(output_dir, exist_ok=True)

    manager = mp.Manager()
    lock = manager.Lock()
    counter = manager.Value('i', 0)

    with mp.Pool(num_workers) as pool:
        pool.starmap(process_file, [(file, llama_tokenizer, llama_model, output_dir, lock, counter) for file in embedding_files])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate audio embeddings to text using LLaMA in parallel.")
    parser.add_argument("embeddings_dir", type=str, help="Directory containing the saved embeddings files")
    parser.add_argument("output_dir", type=str, help="Directory to save the translated text files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    main(args.embeddings_dir, args.output_dir, args.num_workers)


