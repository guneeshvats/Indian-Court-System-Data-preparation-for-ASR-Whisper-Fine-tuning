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

# Example usage:
convert_mp3_to_wav_with_ffmpeg('/Users/guneeshvats/Desktop/Adalat AI assignment/court_audio_files', 'court_audio_files_wav')
