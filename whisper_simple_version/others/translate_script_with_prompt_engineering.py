import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Llama model and tokenizer
model_path = "/data/akshat/models/Meta-Llama-3-8B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    logging.info(f"Model {model_path} loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model {model_path}: {e}")
    raise

def translate_text(text, max_new_tokens=30):
    try:
        prompt = f"Translate the following French text to English:\n\n{text}\n\nTranslation:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the English part after the "Translation:" prompt
        translation = translation.split("Translation:")[-1].strip()
        return translation
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return ""

def save_translations(translations, output_file):
    try:
        with open(output_file, "w") as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
        logging.info(f"Translations saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving translations: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate transcriptions from French to English using Llama model")
    parser.add_argument("input_file", type=str, help="Path to the JSON file containing transcriptions")
    parser.add_argument("output_file", type=str, help="Path to save the translated JSON file")
    args = parser.parse_args()

    # Load the transcriptions
    logging.info(f"Loading transcriptions from {args.input_file}")
    try:
        with open(args.input_file, "r") as f:
            transcriptions = json.load(f)
        logging.info(f"Loaded {len(transcriptions)} transcriptions")
    except Exception as e:
        logging.error(f"Error loading transcriptions: {e}")
        raise

    translations = []
    for index, item in enumerate(tqdm(transcriptions, desc="Translating texts")):
        try:
            text = item["transcribed_text"]
            logging.info(f"Translating text: {text}")
            translated_text = translate_text(text)
            logging.info(f"Translated Text: {translated_text}")
            translations.append({"index": item["index"], "transcribed_text": text, "translated_text": translated_text})

            # Save intermediate results every 5 iterations
            if (index + 1) % 5 == 0:
                logging.info(f"Saving intermediate results to {args.output_file}")
                save_translations(translations, args.output_file)
        except KeyError as e:
            logging.error(f"Missing key in transcription item: {e}")
        except Exception as e:
            logging.error(f"Error translating text: {e}")

    # Save the final translations
    logging.info(f"Saving final results to {args.output_file}")
    save_translations(translations, args.output_file)
    logging.info("Translation process completed.")

                 
"""
Translating texts:   0%|                                                                                                                                                     | 0/16159 [00:00<?, ?it/s]INFO:root:Translating text:  Ce dernier a évolué tout au long de l'histoire romaine.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
INFO:root:Translated Text:  Ce dernier a évolué tout au long de l'histoire romaine. Il a été utilisé pour désigner les soldats de l'armée romaine, puis les soldats de l'armée byzantine, puis les soldats de l'armée byzantine et des armées des États successeurs de l'Empire byzantin. Il a également été utilisé pour désigner les soldats de l'armée ottomane, puis les soldats de l'armée turque moderne.

## Histoire

### Antiquité

Le terme « spatharios » est attesté pour la première fois dans les sources littéraires grecques au IVe siècle. Il est utilisé pour désigner les soldats de l'armée romaine. Au Ve siècle, le terme est utilisé pour désigner les soldats de l'armée byzantine. Au VIe siècle, le terme est utilisé pour désigner les soldats de l
Translating texts:   0%|  

(myenv) daisysxm76@lingua:/data/daisysxm76/speechtospeech$ CUDA_VISIBLE_DEVICES=4 python3 /data/daisysxm76/speechtospeech/whisper/whisper_simple_version/translate_script_with_prompt_engineering.py /data/daisysxm76/speechtospeech/dataset_fr_en/transcriptions.json /data/daisysxm76/speechtospeech/dataset_fr_en/translations.json
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.65s/it]
INFO:root:Model /data/akshat/models/Meta-Llama-3-8B loaded successfully.
INFO:root:Loading transcriptions from /data/daisysxm76/speechtospeech/dataset_fr_en/transcriptions.json
INFO:root:Loaded 16159 transcriptions
Translating texts:   0%|                                                                                                                              | 0/16159 [00:00<?, ?it/s]INFO:root:Translating text:  Ce dernier a évolué tout au long de l'histoire romaine.
INFO:root:Translated Text: This last one has evolved throughout the history of Rome.

## See also

* Wikibooks: French
* Wikibooks: English
Translating texts:   0%|                                                                                                                  | 1/16159 [00:53<241:30:38, 53.81s/it]INFO:root:Translating text:  Son actionnaire majoritaire est le conseiller territorial de Saint-Pierre-et-Miquelon.
INFO:root:Translated Text: The majority shareholder is the territorial councillor of Saint-Pierre-et-Miquelon.

## See also

* French language
* French language tests
* French language tests for beginners
* French language tests for intermediate learners
* French language tests for advanced learners
* French language tests for native speakers
* French language tests for teachers
* French language tests for students
* French language tests for children
* French language tests for adults
* French language tests for seniors
* French language tests for business
* French language tests for tourism
* French language tests for travel
* French language tests for work
* French language tests for study
* French language tests for school
* French language tests for university
* French language tests for college
* French language tests for high school
* French language tests for elementary school
* French language tests for middle school
* French language tests for preschool
* French language tests for kindergarten
* French language tests for primary school
* French language
Translating texts:   0%|                                                                                                                 | 2/16159 [05:47<874:26:37, 194.84s/it]INFO:root:Translating text:  Ce site contient 4 tombeaux de la dynastie Hacheménide et 7 des Sassanides.
INFO:root:Translated Text: This site contains 4 tombeaux of the Hacheménide dynasty and 7 of the Sassanides.

## See also

* List of ancient Iranian tribes
Translating texts:   0%|    

"""
"""
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Llama model and tokenizer
model_path = "/data/akshat/models/Meta-Llama-3-8B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    logging.info(f"Model {model_path} loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model {model_path}: {e}")
    raise

def translate_text(text, max_length=200):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_length)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return ""

def save_translations(translations, output_file):
    try:
        with open(output_file, "w") as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
        logging.info(f"Translations saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving translations: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate transcriptions from French to English using Llama model")
    parser.add_argument("input_file", type=str, help="Path to the JSON file containing transcriptions")
    parser.add_argument("output_file", type=str, help="Path to save the translated JSON file")
    args = parser.parse_args()

    # Load the transcriptions
    logging.info(f"Loading transcriptions from {args.input_file}")
    try:
        with open(args.input_file, "r") as f:
            transcriptions = json.load(f)
        logging.info(f"Loaded {len(transcriptions)} transcriptions")
    except Exception as e:
        logging.error(f"Error loading transcriptions: {e}")
        raise

    translations = []
    for index, item in enumerate(tqdm(transcriptions, desc="Translating texts")):
        try:
            text = item["transcribed_text"]
            logging.info(f"Translating text: {text}")
            translated_text = translate_text(text)
            logging.info(f"Translated Text: {translated_text}")
            translations.append({"index": item["index"], "transcribed_text": text, "translated_text": translated_text})

            # Save intermediate results every 5 iterations
            if (index + 1) % 5 == 0:
                logging.info(f"Saving intermediate results to {args.output_file}")
                save_translations(translations, args.output_file)
        except KeyError as e:
            logging.error(f"Missing key in transcription item: {e}")
        except Exception as e:
            logging.error(f"Error translating text: {e}")

    # Save the final translations
    logging.info(f"Saving final results to {args.output_file}")
    save_translations(translations, args.output_file)
    logging.info("Translation process completed.")

"""

