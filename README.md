# ASR Fine-Tuning Project with Whisper

## Project Overview
This project aims to fine-tune OpenAI’s Whisper ASR model on Supreme Court hearing transcription data. The data is preprocessed to align audio segments with their respective dialogues and speaker diarization. The Whisper model is then fine-tuned to improve its transcription accuracy on the court hearing data.

## Project Structure
- **data/**: Contains the preprocessed audio files and transcripts.
- **models/**: Stores the fine-tuned Whisper model.
- **scripts/**: Python scripts for data processing, diarization, and fine-tuning.
- **report.md**: A detailed report of the project.
- **README.md**: This file.

## Steps to Run the Project

### 1. Data Preparation
We started by preparing the dataset:
- Remove silence from audio files.
- Perform speaker diarization using Pyannote.audio.
- Align the speaker diarization with the corresponding transcripts.
- Split the original audio based on the aligned segments.

### 2. Fine-tune Whisper Model
After preparing the data, follow these steps to fine-tune the Whisper model:
1. Install the necessary libraries:
   ```bash
   pip install transformers datasets torchaudio librosa


## Folder Structure 
```ASR_Fine_Tuning_Project/
├── data/                       
│   ├── audio/                  
│   ├── transcripts/            
│   ├── dataset.csv             
├── models/                     
│   └── whisper/                
│       ├── config.json         
│       ├── pytorch_model.bin   
│       └── tokenizer.json      
├── scripts/                    
│   ├── diarization.py          
│   ├── prepare_dataset.py      
│   └── fine_tune_whisper.py    
├── output/                     
│   └── evaluation.json         
├── report.md                   
├── README.md                   
└── requirements.txt       
```