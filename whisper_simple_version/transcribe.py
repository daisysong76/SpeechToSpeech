
# transcribe.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import resampy
import soundfile as sf

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")

def transcribe_audio(file_path):
    try:
        # Load audio file without specifying additional parameters
        audio_input, sample_rate = sf.read(file_path)

        # Resample to 16000 Hz if the sample rate is different
        if sample_rate != 16000:
            audio_input = resampy.resample(audio_input, sample_rate, 16000)
            sample_rate = 16000

        # Preprocess the audio input
        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to("cuda")

        # Perform inference
        generated_ids = model.generate(input_features)

        # Decode the transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None