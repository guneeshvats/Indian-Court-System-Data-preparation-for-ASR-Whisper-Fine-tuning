# To know how many CPU cores you have, run this command - "sysctl -n hw.ncpu" and the adjust the parameter num_workers based on that
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from concurrent.futures import ProcessPoolExecutor

def remove_silence_from_single_file(audio_file, input_folder, output_folder):
    if audio_file.endswith(".wav"):
        sound = AudioSegment.from_wav(os.path.join(input_folder, audio_file))
        chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=-40)
        output_path = os.path.join(output_folder, audio_file)

        # Concatenate chunks to create a cleaned file
        output_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            output_audio += chunk

        output_audio.export(output_path, format="wav")
        print(f"Processed and saved: {output_path}")

def remove_silence_from_audio_parallel(input_folder, output_folder, num_workers=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(remove_silence_from_single_file, audio_file, input_folder, output_folder) for audio_file in audio_files]

        for future in futures:
            future.result()  # Wait for each file to be processed

# Example usage:
remove_silence_from_audio_parallel('path_to_wav_folder', 'output_clean_wav_folder', num_workers=4)
