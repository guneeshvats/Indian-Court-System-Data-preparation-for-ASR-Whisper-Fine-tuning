############################################################################################################################
############################################################################################################################
'''
Purpose : Perform speaker diarization on multiple audio files using the pre-trained pyannote speaker diarization model.

Steps : 

1. Authentication: Use the Hugging Face token to authenticate and load the pre-trained pipeline.
2. Process Each Audio File: Runs the speaker diarization model on each audio file in the input folder.
3. Save Diarization Results: Saves the diarization output in the RTTM format for each audio file in the specified output folder.

Replace Paths and Tokens : 

no_silence_wav_folder = "path_to_input_audio_files"
output_folder = "path_to_save_diarization_results"
hf_token = "your_huggingface_token"

Code written by: Guneesh Vats
Date: 25th Sept, 2024
'''
############################################################################################################################
############################################################################################################################

from pyannote.audio import Pipeline
from pyannote.audio import Audio
from transformers import AutoModel
from huggingface_hub import login
import os


def diarize_speakers(input_audio_folder, diarization_output_folder, hf_token):
    # Load the pre-trained pipeline using the token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    
    if not os.path.exists(diarization_output_folder):
        os.makedirs(diarization_output_folder)

    for audio_file in os.listdir(input_audio_folder):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(input_audio_folder, audio_file)
            diarization = pipeline(audio_path)

            # Save diarization results in RTTM format
            rttm_path = os.path.join(diarization_output_folder, audio_file.replace(".wav", ".rttm"))
            with open(rttm_path, "w") as rttm_file:
                diarization.write_rttm(rttm_file)
            print(f"Speaker diarization done and saved at: {rttm_path}")



no_silence_wav_folder = "/Users/guneeshvats/Desktop/Adalat AI assignment/Approach_2/no_silence_audio_folder"
output_folder = "/Users/guneeshvats/Desktop/Adalat AI assignment/Approach_2/output_diarization_folder"
hf_token = "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"
diarize_speakers(no_silence_wav_folder, output_folder, hf_token)





