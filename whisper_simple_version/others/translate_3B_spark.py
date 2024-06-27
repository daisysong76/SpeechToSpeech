import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType
import pandas as pd

# Initialize Spark session with increased memory configuration
spark = SparkSession.builder \
    .appName("TranslationProcessing") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Llama model and tokenizer
model_path = "/data/akshat/models/Meta-Llama-3-8B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    logging.info(f"Model {model_path} loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model {model_path}: {e}")
    raise

def translate_text(text, max_new_tokens=50):
    try:
        prompt = f"Translate the following French text to English:\n\n{text}\n\nTranslation:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = translation.split("Translation:")[-1].strip().split('\n')[0]
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

    # Convert transcriptions to DataFrame
    df = pd.DataFrame(transcriptions)
    spark_df = spark.createDataFrame(df).repartition(20)  # Repartitioning to reduce memory usage

    @pandas_udf(StringType())
    def translate_udf(text_series: pd.Series) -> pd.Series:
        return text_series.apply(translate_text)

    translations = []
    for i in range(0, spark_df.count(), 1000):
        batch_df = spark_df.limit(1000).toPandas()
        batch_df['translated_text'] = batch_df['transcribed_text'].apply(translate_text)
        translated_batch = batch_df.to_dict(orient="records")
        translations.extend(translated_batch)
        # Save intermediate results
        save_translations(translations, args.output_file)

    # Save the final translations
    logging.info(f"Saving final results to {args.output_file}")
    save_translations(translations, args.output_file)
    logging.info("Translation process completed.")




