
import pandas as pd
import argparse

def check_dataset(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        print("Column names in the dataset:")
        print(df.columns)
        print("\nFirst few rows of the dataset:")
        print(df.head())
    except Exception as e:
        print(f"Error reading the dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset contents")
    parser.add_argument("dataset", type=str, help="Path to the dataset TSV file")
    args = parser.parse_args()
    
    check_dataset(args.dataset)
