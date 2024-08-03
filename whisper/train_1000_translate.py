import pandas as pd
import sys

def match_and_save_translations(first_1000_tsv: str, translations_tsv: str, output_tsv: str):
    # Read the first 1000 items
    df_first_1000 = pd.read_csv(first_1000_tsv, sep='\t')
    print(f"Columns in first 1000 TSV: {df_first_1000.columns}")
    print(f"First few rows in first 1000 TSV:\n{df_first_1000.head()}")
    
    # Read the translations
    df_translations = pd.read_csv(translations_tsv, sep='\t')
    print(f"Columns in translations TSV: {df_translations.columns}")
    print(f"First few rows in translations TSV:\n{df_translations.head()}")
    
    # Check for the presence of 'translation' in the translation file
    if 'translation' not in df_translations.columns or 'path' not in df_translations.columns:
        raise KeyError("'translation' or 'path' not found in translations TSV file")

    # Merge on 'path' column
    if 'path' in df_first_1000.columns and 'path' in df_translations.columns:
        df_first_1000_translations = df_first_1000.merge(df_translations, on='path')
        print(f"Columns after merge: {df_first_1000_translations.columns}")
        print(f"First few rows after merge:\n{df_first_1000_translations.head()}")
    else:
        raise KeyError("'path' not found in one of the TSV files")

    # Select relevant columns
    if 'path' in df_first_1000_translations.columns and 'translation' in df_first_1000_translations.columns and 'split' in df_first_1000_translations.columns:
        df_first_1000_translations = df_first_1000_translations[['path', 'translation', 'split']]
    else:
        raise KeyError("'path' or 'translation' or 'split' not found after merging")

    # Save to new TSV file
    df_first_1000_translations.to_csv(output_tsv, sep='\t', index=False)
    print(f"Saved matched items to {output_tsv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 train_1000_translate.py <first_1000_tsv> <translations_tsv> <output_tsv>")
        sys.exit(1)
    
    first_1000_tsv = sys.argv[1]
    translations_tsv = sys.argv[2]
    output_tsv = sys.argv[3]

    match_and_save_translations(first_1000_tsv, translations_tsv, output_tsv)



# CUDA_VISIBLE_DEVICES=1  python3 /data/daisysxm76/speechtospeech/whisper/whisper/train_1000_translate.py /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train_first_1000.tsv /data/daisysxm76/speechtospeech/dataset_fr_en/covost_v2.fr_en.tsv /data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/train_first1000_translate.tsv