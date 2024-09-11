from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def remove_silence_from_audio(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".wav"):
            sound = AudioSegment.from_wav(os.path.join(input_folder, audio_file))
            chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=-40)
            output_path = os.path.join(output_folder, audio_file)
            
            # Concatenate chunks to create a cleaned file without silence
            output_audio = AudioSegment.silent(duration=0)
            for chunk in chunks:
                output_audio += chunk
            
            output_audio.export(output_path, format="wav")
            print(f"Processed and saved: {output_path}")

# Example usage:


# Example usage:
audio_folder_wav = "/Users/guneeshvats/Desktop/Adalat AI assignment/Approach_2/court_audio_files_wav"
no_silence_audio_folder = "no_silence_audio_folder"
remove_silence_from_audio(audio_folder_wav, no_silence_audio_folder)
print("All files are processed and silence is removed!")
