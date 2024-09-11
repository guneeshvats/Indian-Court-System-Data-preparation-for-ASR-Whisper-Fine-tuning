# Project Report: ASR Model Fine-tuning with Whisper

## Introduction
This project aims to fine-tune a pre-trained Automatic Speech Recognition (ASR) model, specifically OpenAI's Whisper model, using data from Supreme Court Hearing Transcriptions. The key goal is to improve the model's performance on transcribing legal proceedings accurately.

## Steps Overview
We followed a structured approach, including data preprocessing, aligning speaker diarization with transcripts, preparing the ASR dataset, and finally fine-tuning the Whisper model.

### 1. Data Preprocessing
In the first step, we processed the raw audio data by removing silence, identifying speaker changes, and aligning the dialogues with the respective audio segments. This step is important because it helps in breaking down long court hearings into meaningful segments which are easier for the ASR model to handle. 

### 2. Speaker Diarization and Transcript Alignment
We used diarization techniques to separate the audio into speaker segments. The aligned JSON output mapped the speaker's dialogues to the correct timestamps, ensuring that each dialogue corresponds to the correct speaker and part of the audio. This step helped reduce errors caused by overlapping or misidentified speakers, thereby improving transcription accuracy.

### 3. Forced Alignment Consideration
Although forced alignment (aligning every word in the transcript with the exact timestamp in the audio) could provide a more granular alignment, it was deemed unnecessary for our fine-tuning task. Segment-level alignment is sufficient for ASR fine-tuning, and forced alignment would have added unnecessary complexity without significant benefits.

### 4. Data Preparation for ASR Fine-tuning
We processed the audio files based on the speaker diarization JSON and split the original audio into segments. The data was then formatted into the standard ASR format, with each segment paired with its corresponding transcript. This step was crucial to make the dataset compatible with the Whisper model.

### 5. Fine-tuning Whisper Model
We fine-tuned the pre-trained Whisper model using Hugging Face’s `transformers` library. We chose Whisper because it is a state-of-the-art ASR model, particularly strong for multilingual and noisy environments, making it well-suited for the legal domain.

- **Reason for Using Whisper**: Whisper's architecture is well-suited for handling long audio segments and multilingual transcripts, making it a strong choice for court hearings.

We configured the training process with appropriate hyperparameters, including batch size, learning rate, and evaluation strategy.

### 6. Model Evaluation
After fine-tuning, we evaluated the model using the Word Error Rate (WER) metric. WER provides a good indicator of the model's performance by measuring how well the predicted transcripts match the actual transcripts. We also analyzed areas for potential future improvements, such as better handling of overlapping speech.

## Conclusion
The project successfully fine-tuned the Whisper model to improve transcription accuracy for Supreme Court hearing data. By using speaker diarization and segment-based alignment, we reduced errors related to overlapping speech and complex dialogues. Forced alignment was not needed in this context as the segment-based approach was sufficient and less computationally intensive.
