# New class by using api_client
import os
import openai

openai.api_key = 'your-api-key'

def transcribe_audio(file_path, language="en"):
    """
    Transcribe an audio file using Whisper API.

    Args:
    - file_path (str): Path to the audio file.
    - language (str): Language code for transcription (default is "en").

    Returns:
    - str: Transcribed text.
    """
    with open(file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            language=language
        )
    return response['text']

def transcribe_audio_batch(directory, language="en"):
    """
    Transcribe multiple audio files in a directory using Whisper API.

    Args:
    - directory (str): Path to the directory containing audio files.
    - language (str): Language code for transcription (default is "en").

    Returns:
    - dict: Dictionary with filenames as keys and transcribed text as values.
    """
    transcriptions = {}
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            try:
                transcriptions[filename] = transcribe_audio(file_path, language)
            except Exception as e:
                transcriptions[filename] = f"Error: {e}"
    return transcriptions
