import pandas as pd
import argparse

def save_first_1000_items(input_tsv: str, output_tsv: str):
    """
    Reads the first 1000 items from the input TSV file and saves them into the output TSV file.
    
    Args:
    input_tsv (str): Path to the input TSV file.
    output_tsv (str): Path to the output TSV file.
    """
    # Read the input TSV file
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Select the first 1000 items
    df_1000 = df.head(2000)
    
    # Save the first 1000 items to the output TSV file
    df_1000.to_csv(output_tsv, sep='\t', index=False)
    print(f"Saved first 1000 items to {output_tsv}")

def main():
    parser = argparse.ArgumentParser(description="Extract the first 1000 items from a TSV file and save to a new TSV file.")
    parser.add_argument("input_tsv", type=str, help="Path to the input TSV file")
    parser.add_argument("output_tsv", type=str, help="Path to the output TSV file")
    
    args = parser.parse_args()
    
    save_first_1000_items(args.input_tsv, args.output_tsv)

if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=1  python3 /data/daisysxm76/speechtospeech/whisper/whisper/train_first100.py  /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train.tsv /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train_first_1000.tsv
