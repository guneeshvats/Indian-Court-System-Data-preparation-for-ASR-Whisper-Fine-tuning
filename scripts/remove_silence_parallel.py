############################################################################################################################
############################################################################################################################
'''
Purpose : Remove silence from multiple audio files in a directory using parallel processing to speed up the operation.

Steps : 

1. Remove Silence: Splits each audio file into chunks based on silence and concatenates them to form a cleaned audio file.
2. Applying some pre-processing steps like - remvoing bg noise using LPF and HPF and normalized the audio files
   (Filters are working better than Spectral Subtraction)
3. Parallel Processing: Uses the `ProcessPoolExecutor` to run the silence removal in parallel across multiple CPU cores.
4. Save Cleaned Files: Exports the cleaned audio files to the specified output directory.

Replace Paths : 

input_folder = "path_to_input_wav_files"
output_folder = "path_to_output_cleaned_wav_files"

To know how many CPU cores you have, run this command - "sysctl -n hw.ncpu" and the adjust the parameter num_workers based on that

Code written by: Guneesh Vats
Date: 25th Sept, 2024
'''
############################################################################################################################
############################################################################################################################


from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from concurrent.futures import ProcessPoolExecutor

# Function to apply preprocessing steps such as silence removal, filtering, and normalization
def preprocess_audio(audio_segment, high_pass_freq=300, low_pass_freq=3000):
    # Apply High-Pass and Low-Pass Filters
    filtered_audio = audio_segment.high_pass_filter(high_pass_freq).low_pass_filter(low_pass_freq)
    # Normalize audio for uniform volume levels
    normalized_audio = filtered_audio.normalize()
    return normalized_audio

def remove_silence_and_preprocess_single_file(audio_file, input_folder, output_folder, high_pass_freq=300, low_pass_freq=3000):
    if audio_file.endswith(".wav"):
        sound = AudioSegment.from_wav(os.path.join(input_folder, audio_file))
        # Remove silence from audio
        chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=-40)

        # Concatenate chunks to create a cleaned file and apply preprocessing
        output_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            # Apply additional preprocessing (filtering and normalization) on each chunk
            processed_chunk = preprocess_audio(chunk, high_pass_freq, low_pass_freq)
            output_audio += processed_chunk

        # Export the processed audio to the output folder
        output_path = os.path.join(output_folder, audio_file)
        output_audio.export(output_path, format="wav")
        print(f"Processed and saved: {output_path}")

def remove_silence_and_preprocess_audio_parallel(input_folder, output_folder, num_workers=4, high_pass_freq=300, low_pass_freq=3000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(remove_silence_and_preprocess_single_file, audio_file, input_folder, output_folder, high_pass_freq, low_pass_freq) for audio_file in audio_files]

        for future in futures:
            future.result()  # Wait for each file to be processed

# Example usage
remove_silence_and_preprocess_audio_parallel('path_to_wav_folder', 'output_clean_wav_folder', num_workers=4)
