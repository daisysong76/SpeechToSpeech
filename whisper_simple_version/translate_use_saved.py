import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/data/akshat/models/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("/data/akshat/models/Meta-Llama-3-8B")

def translate_text_with_prompt(text):
    prompt = f"Translate the following French text to English: '{text}'"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract the translated part
    if "Translate the following French text to English:" in translation:
        translation = translation.split("Translate the following French text to English:")[-1].strip()
        translation = translation.split("English:")[-1].strip()
    # Get everything before the first "\n"
    translation = translation.split("\n")[0].strip()
    return translation

def save_translations(translations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate transcriptions using Llama-3-8B with prompt engineering")
    parser.add_argument("transcriptions_file", type=str, help="Path to the transcriptions JSON file")
    parser.add_argument("output_file", type=str, help="Path to save the translated JSON file")
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load transcriptions
    logging.info(f"Loading transcriptions from {args.transcriptions_file}")
    with open(args.transcriptions_file, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)

    logging.info(f"Loaded {len(transcriptions)} transcriptions")

    # Translate transcriptions
    translations = []
    for index, item in enumerate(tqdm(transcriptions, desc="Translating texts")):
        text = item["transcribed_text"]
        try:
            logging.info(f"Translating text: {text}")
            translated_text = translate_text_with_prompt(text)
            logging.info(f"Translated Text: {translated_text}")
            translations.append({
                "index": item["index"],
                "transcribed_text": text,
                "translated_text": translated_text
            })
        except Exception as e:
            logging.error(f"Error translating text {text}: {e}")
            translations.append({
                "index": item["index"],
                "transcribed_text": text,
                "translated_text": None
            })

        # Save intermediate results every 5 iterations
        if index % 5 == 0:
            logging.info(f"Saving intermediate results to {args.output_file}")
            save_translations(translations, args.output_file)

    # Save the final translations
    logging.info(f"Saving final results to {args.output_file}")
    save_translations(translations, args.output_file)
    logging.info("Translation process completed.")

