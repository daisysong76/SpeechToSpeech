
# new code

# import whisper
# import os

# def transcribe_audio_file(file_path):
#     # Load the Whisper model
#     model = whisper.load_model("base")  # Choose the appropriate model size

#     # Transcribe the audio file
#     result = model.transcribe(file_path)

#     # Return the transcription
#     return result["text"]

# def main():
#     # Path to the audio file
#     audio_file_path = "audio_files/example.wav"

#     # Check if the file exists
#     if not os.path.exists(audio_file_path):
#         print(f"Audio file not found: {audio_file_path}")
#         return

#     # Transcribe the audio file
#     transcription = transcribe_audio_file(audio_file_path)

#     # Print the transcription
#     print("Transcription:")
#     print(transcription)

# if __name__ == "__main__":
#     main()




# old stuff the rest

from .others.transcribe_old import cli


cli()
