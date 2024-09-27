############################################################################################################################
############################################################################################################################
'''
Purpose : Prepare Automatic Speech Recognition (ASR) data by splitting audio files based on timestamps from JSON files 
          and creating corresponding transcript files.

Steps : 

1. Split Audio: Splits the audio files into segments using start and end times specified in JSON files.
2. Create Transcript Files: Saves each transcript segment in a separate text file corresponding to the audio segment.
3. Generate Metadata: Creates a metadata file containing information about the audio segments, transcripts, and speakers.

Replace Paths : 

json_folder = "path_to_json_files"
audio_folder = "path_to_audio_files which are silence removed and processed with noise removal"
output_folder = "path_to_save_ASR_data"

Code written by: Guneesh Vats
Date: 25th Sept, 2024
'''
############################################################################################################################
############################################################################################################################


import os
import json
import subprocess

def split_audio(audio_file, start_time, end_time, output_file):
    """Splits audio file based on start and end times using ffmpeg."""
    command = ['ffmpeg', '-i', audio_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', output_file]
    subprocess.run(command)

def prepare_asr_data(json_folder, audio_folder, output_folder):
    """Prepares ASR dataset from aligned JSON files and audio files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metadata = []

    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            base_name = json_file.replace(".json", "")
            audio_file = os.path.join(audio_folder, base_name + ".wav")
            
            # Load the JSON file
            json_path = os.path.join(json_folder, json_file)
            with open(json_path, 'r') as f:
                aligned_data = json.load(f)

            for idx, segment in enumerate(aligned_data):
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                transcript = segment["transcript"]

                # Create file names for the audio segment and transcript
                audio_output = os.path.join(output_folder, f"{base_name}_segment_{idx}.wav")
                transcript_output = os.path.join(output_folder, f"{base_name}_segment_{idx}.txt")

                # Split the audio
                split_audio(audio_file, start_time, end_time, audio_output)

                # Write the transcript
                with open(transcript_output, 'w') as transcript_file:
                    transcript_file.write(transcript)

                # Append to metadata
                metadata.append({
                    "audio": audio_output,
                    "transcript": transcript_output,
                    "speaker": segment["real_speaker"]
                })

    # Save metadata to a JSON or CSV file
    metadata_file = os.path.join(output_folder, 'metadata.json')
    with open(metadata_file, 'w') as mf:
        json.dump(metadata, mf, indent=4)

# Example usage:
json_folder = 'final_output_json'
audio_folder = 'original_audio_folder'
output_folder = 'ASR_data'

prepare_asr_data(json_folder, audio_folder, output_folder)
