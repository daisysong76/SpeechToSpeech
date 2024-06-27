

import json
import csv

def read_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_tsv_file(tsv_path):
    tsv_data = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if not tsv_data:  # Print columns on first row
                print(f"TSV Columns: {row.keys()}")
            tsv_data[row['path']] = row['translation']
    return tsv_data

def match_audio_files(json_data, tsv_data):
    matched_results = []
    for item in json_data:
        audio_file = item['audio_file']
        if audio_file in tsv_data:
            matched_results.append({
                "index": item["index"],
                "audio_file": audio_file,
                "transcribed_text": item["transcribed_text"],
                "translated_text": item["translated_text"],
                "original_transcription": item["original_transcription"],
                "tsv_translation": tsv_data[audio_file]
            })
        else:
            print(f"Audio file {audio_file} not found in TSV data.")
    return matched_results

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Define paths
json_path = '/data/daisysxm76/speechtospeech/dataset_fr_en/translations_3B.json'
tsv_path = '/data/daisysxm76/speechtospeech/dataset_fr_en/covost_v2.fr_en.tsv'
output_path = '/data/daisysxm76/speechtospeech/matched_results.json'

# Read input files
json_data = read_json_file(json_path)
tsv_data = read_tsv_file(tsv_path)

# Match audio files
matched_results = match_audio_files(json_data, tsv_data)

# Save matched results
save_to_json(matched_results, output_path)

print(f"Matched results have been saved to {output_path}")
