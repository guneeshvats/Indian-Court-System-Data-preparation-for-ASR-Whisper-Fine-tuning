# ASR Fine-Tuning Project with Whisper

## Project Overview
This project aims to fine-tune OpenAI’s Whisper ASR model on Supreme Court hearing transcription data. The data is preprocessed to align audio segments with their respective dialogues and speaker diarization. The Whisper model is then fine-tuned to improve its transcription accuracy on the court hearing data.

## Project Structure
- **data/**: Contains the preprocessed audio files, transcripts, .
- **models/**: Stores the fine-tuned Whisper model.
- **scripts/**: Python scripts for data processing, `silence removal`, `diarization`, `alignment` and `fine-tuning`.
- **report.md**: A detailed report of the project
- **README.md**: This file.


To check how many CPU cores you have on your System Run this command :
   ```bash 
   sysctl -n hw.ncpu
   ```
This will be helpful for the parameter - `num_workers` in `silence_removal_parallel.py` file from scripts folder 

## Steps to Run the Project

### 1. Data Preparation
We started by preparing the dataset given to us in `dataset.csv` file:
- Download the audio files and pdf transcripts given through the links in the csv file.
- Run the `mp3_to_wav.py` file to convert the extension from `.mp3` to `.wav`
- To remove silence from audio files run the - `remove_silence.py` file
- Perform speaker diarization using `Pyannote.audio` model from the `diarization.py` file. You would need to use your hugging face token for this file for a gated model from hugging face and also download the model using the follwing commands in your terminal. 
Download the Diarization model
   ```bash
   huggingface-cli download pyannote/speaker-diarization
   huggingface-cli download pyannote/segmentation
- Align the speaker diarization with the corresponding transcripts and generate the json files for each pair of transcript and audio files using the file in the scripts folder - `Alignment.py`. All those json files will be stored in `data/text_aligned_json_files`
- Split the original audio based on the aligned segments using the file - `Data_prep_for_ASR.py`. 

### 2. Fine-tune Whisper Model
After preparing the data, follow these steps to fine-tune the Whisper model:
1. Install the necessary libraries:
   ```bash
   pip install transformers datasets torchaudio librosa
2. Run the file - `fine_tune_whisper.py`


## Folder Structure 
```ASR_Fine_Tuning_Project/
├── data/                       
│   ├── audio/                  
│   ├── transcripts/            
│   ├── dataset.csv
|   ├── text_aligned_json_files/
|   ├── Diarized_Files/
|   ├── Audio_Segments_transcripts_splits/                                     
├── models/                     
│   └── whisper/                
│       ├── config.json         
│       ├── pytorch_model.bin   
│       └── tokenizer.json      
├── scripts/                    
│   ├── mp3_to_wav.py         
│   ├── remove_silence.py
│   ├── remove_silence_parallel.py
|   ├── diarization.py 
|   ├── Alignment.py
|   ├── Data_prep_for_ASR.py
│   └── fine_tune_whisper.py
├── output/                     
│   └── evaluation.json         
├── Report.md                   
├── README.md                   
└── requirements.txt
```

### Step-by-Step Process
   
1. **Data Preprocessing**:
   - Silence is removed from raw audio using `prepare_dataset.py`.
   - Speaker diarization is performed with `diarization.py`, which splits the audio by speaker and outputs aligned JSON files.

2. **Transcript Alignment**:
   - Transcripts are aligned with the diarized audio segments using `diarization.py`, generating JSON files with speaker dialogue and timestamps.

3. **Dataset Preparation**:
   - `prepare_dataset.py` splits the original audio based on diarization segments and prepares the dataset for ASR fine-tuning, ensuring each audio segment is paired with a transcript.

4. **Fine-tuning Whisper**:
   - The `fine_tune_whisper.py` script fine-tunes the pre-trained Whisper model using the prepared dataset, adjusting hyperparameters like batch size and learning rate.

5. **Evaluation**:
   - The fine-tuned model is evaluated using Word Error Rate (WER) in `fine_tune_whisper.py`, with results saved to `output/evaluation.json`.

