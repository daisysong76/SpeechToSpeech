# translate.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat")

def translate_text(text, source_lang="fr", target_lang="en"):
    try:
        # Prepare the input text with language tokens if required by the model
        inputs = tokenizer(f"<s> {source_lang} {text} </s> {target_lang}", return_tensors="pt")
        outputs = model.generate(**inputs)
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translation[0]
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text using a translation model")
    parser.add_argument("text", type=str, help="Text to translate")
    parser.add_argument("--source_lang", type=str, default="fr", help="Source language (default: 'fr')")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (default: 'en')")
    args = parser.parse_args()

    result = translate_text(args.text, args.source_lang, args.target_lang)
    if result:
        print(f"Translated Text: {result}")
    else:
        print("Failed to translate text.")
