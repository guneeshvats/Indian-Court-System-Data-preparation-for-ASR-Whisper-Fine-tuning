############################################################################################################################
############################################################################################################################
'''
Purpose : Convert all MP3 files in a directory to WAV format using FFmpeg.

Steps : 

1. Create Output Directory: Creates a new folder if the specified output folder does not exist.
2. Iterate through Files: Loops through each file in the input directory and checks if it is an MP3 file.
3. Convert MP3 to WAV: Uses FFmpeg to convert each MP3 file to WAV format and saves it in the specified output directory.

Replace Paths : 

input_folder = "path_to_input_folder_containing_mp3_files"
output_folder = "path_to_output_folder_for_wav_files"

Code written by: Guneesh Vats
Date: 30th Aug, 2024
'''
############################################################################################################################
############################################################################################################################

import os
import subprocess

def convert_mp3_to_wav_with_ffmpeg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".mp3"):
            input_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, audio_file.replace(".mp3", ".wav"))
            
            try:
                # Use FFmpeg to convert MP3 to WAV
                subprocess.run(['ffmpeg', '-i', input_path, output_path], check=True)
                print(f"Converted and saved: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {audio_file}: {e}")

input_folder = "/Users/guneeshvats/Desktop/Adalat AI assignment/court_audio_files"
output_folder = "court_audio_files_wav"
convert_mp3_to_wav_with_ffmpeg(input_folder, output_folder)
