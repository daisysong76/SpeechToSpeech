
import json

def trim_translated_texts(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'translated_text' in item and item['translated_text']:
            item['translated_text'] = item['translated_text'].split('\n')[0].strip()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/translations.json'
    output_file = '/data/daisysxm76/speechtospeech/dataset_fr_en/trimmed_translations.json'
    trim_translated_texts(input_file, output_file)
    print(f"Trimmed translations saved to {output_file}")
