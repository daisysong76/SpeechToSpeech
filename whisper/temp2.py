import os
import logging

audio_file_path = '/data/daisysxm76/speechtospeech/dataset_fr_en/cv-corpus-17.0-2024-03-15/fr/clips/common_voice_fr_17945776.mp3'

if not os.path.exists(audio_file_path):
    print(f"Audio file {audio_file_path} not found.")
else:
    print(f"Audio file {audio_file_path} found.")