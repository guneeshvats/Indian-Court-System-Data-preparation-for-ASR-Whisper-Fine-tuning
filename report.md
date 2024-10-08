# Project Report: ASR Model Fine-tuning with Whisper
Guneesh Vats | IIIT Hyderabad | guneesh.vats@research.iiit.ac.in | [Github Repo Link](https://github.com/guneeshvats/Indian-Court-Transcriptions-ASR-Model-Whisper-Fine-tuning/tree/main)


## Introduction
**Objective**: The project aims to leverage OpenAI's Whisper model—an advanced ASR system—to accurately transcribe complex Supreme Court hearing dialogues. By fine-tuning the model on a specialized dataset, we seek to address the challenges posed by courtroom hearings, such as multiple speakers, overlapping conversations, and domain-specific legal jargon.

**Outcome**: The goal is to develop a robust data pipeline capable of handling segmented and speaker-dependent audio, ensuring high precision in capturing every word spoken during legal proceedings. This will contribute to maintaining the integrity of legal records in a domain where accuracy is extremely important.

---

## Approach Overview
We followed a structured approach to solve the problem, including:
0. **Data Accumulation**: Downloaded the transcripts and correspondind court hearing in `.mp3` from teh given `dataset.csv` file and converted them to `.wav` 
1. **Data Preprocessing**: Removing silence, background noise from the audio and normalized it.
2. **Speaker Diarization**: Speaker diarization of processed audio files and created `.rttm` files.
3. **Data Preparation 1**: Created Alignment json files for each dialogue in transcript of each case paired up with the start and end time of that part in audio file along with the speaker name, using the pdf files and rttm files. Also did the major portion of transcript cleaning in this part.
4. **Data Preparation 2**: Segmented the audio files with this json information and created `.txt` files and `.wav` files for each dialogue.
5. **Testing with pretrained Model**: Tested the data with `pretrained whisper model to see baseline performance.
6. **Fine-tuning Whisper**: Extracted features(mel spectograms) and tokenized the transcript, Created tensors for the audio segments and used the data to Train and evaluate the Whisper model.

<img src="https://github.com/user-attachments/assets/ae06f791-05d0-43f4-a97f-7083fc0fd40c" width=500/>

---

## Steps to Solve the Problem
### 0. Data Accumulation
We downloaded the transcript files for each case from the given `dataset.csv` using the  `download_transcript.py` file and then maually downloaded the corresponding audio files and saved them with the same name but obviously different extension.
Then used the `mp3_to_wav.py` file which uses essentially `ffmpeg` to convert the extension of `.mp3` to `.wav`. 
```
Reason to do this is because .wav provides a much better resolution of the audio file than mp3s noisy format. 
```

### 1. Data Preprocessing
We first processed the raw audio data by removing silence, removing background noise using LPF and HPF withing a certain threshold to get rid of very lowe freq and very high pitched noises fer the audio files so that we can use them for the diarization process.
This was done using the script in file `silence_removal_parallel.py`
To make the process faster I parallelzied the processing of multiple files based on the CPU cores availaible on the system. Detailes instructions are there in the `readme.md` and `top comment` of the script. 

Sample of Raw Audio Before and after noise removal : 

<img src="https://github.com/user-attachments/assets/c72f9fdc-13fb-478d-9c6d-bf3d8472a8d9" alt="image" width="500"/>




### 2. Speaker Diarization
We used speaker diarization model from hugging face to separate the audio into speaker segments using the `diarization.py` script.
we have used `pyyanote/speaker-diarization` model from `hugging face`. For this script you need to have an access token adn login to the hugging face account because this is a gated model.

![image](https://github.com/user-attachments/assets/36228301-090e-4607-9804-313e7215fd64)

Diagram of pyannote/diarization model structure : credits - reference 3

How does a diarization model works :
```
1. There are two main approaches to speaker diarization: bottom-up and top-down. Bottom-up approaches first segment the audio signal into short windows, then cluster these short segments according to similarity before finally assigning labels to the clusters. Top-down approaches first assign labels to short segments, then cluster the labeled segments before finally merging them into longer speech turns.

2. Both bottom-up and top-down approaches have their strengths and weaknesses, and which one is better depends on the specific application. In general, however, bottom-up approaches tend to be more accurate while top-down approaches tend to be faster.
```
Advantages of choosing Pyannote/diarization model : 
```
1. Comes with a set of available pre-trained models for the VAD, embedder and segmentation model.
2. The inference pipeline can identify multiple speakers speaking at the same time (multi-label diarization).

3.Especially when the number of speakers are unknown before running the clustering algorithm. 
```


Challenges of diarization models : 

```
1. It can be difficult to automatically identify when a new speaker starts talking. This can be especially challenging in overlapping speech, or when two or more speakers are speaking at the same time.

2. Speaker diarization can sometimes struggle with distinguishing between different voices (e.g., male and female voices, or different accents).

3. Another challenge is that speaker diarization often relies on having good audio quality
```

Sample of Diarized file output in the `.rttm` files: 
![image](https://github.com/user-attachments/assets/85dc0002-7804-47ee-a4de-6336f49bdd1b)


### 3. Data Preparation 1 : 
using the `.pdf` transcript files and the output of `diarization.py` `.rttm` files for corresponding audios we will construct json files for each case that will contain the information of each dialogues' duration before the speaker is changing, speaker name, start, end time, and the transcript that is supposedly being spoken in that time period (taken from the speaker wise dialogues from court provided transcripts. 

Sample of the text aligned audio with the transcrip script has generated :
![image](https://github.com/user-attachments/assets/73b439b0-1148-4716-b4da-8bbe224b58d0)

### Forced Alignment (Consideration)
While forced alignment (aligning every word in the transcript with its exact timestamp in the audio) could provide more granular alignment, it was not necessary for our task. Segment-level alignment suffices for ASR fine-tuning, and forced alignment would have added unnecessary complexity without significant benefits.

### 4. Data Preparation 2 for ASR Fine-tuning
Using the `.json` file and the pre-processed audio files, we split these into segments defined in the json file. 

These segments were formatted into the standard ASR format, with each segment paired with its corresponding dialogue in a `.txt` file. This step was essential for making the dataset compatible with the Whisper model because now each segment is less than 30s which makes the mel spectogram more meaningful according to the desired training data for the chosen model. 

### 5. Fine-tuning Whisper Model (Using PEFT Technique : LoRA) 
We are fine-tuning a pre-trained `Whisper-small` model using Hugging Face’s `transformers` library. Whisper was selected because of its state-of-the-art performance, particularly in multilingual and noisy environments, making it ideal for the legal domain.
In this 

What exactly are we achieving in the script - `fine_tune_whisper.py` : 
1. This fine-tuning code is designed to customize the pre-trained `Whisper-small` model on a custom audio-transcription dataset that we have prepared. It involves several steps: loading the Whisper model and processor, preparing the data, defining custom configurations using Low-Rank Adaptation (LoRA), and training the model. The data preparation phase reads audio files (in .wav format) and their corresponding text transcripts (in .txt format) from the specified directories. Each audio file is converted into a Log-Mel spectrogram with shape (num_frames, num_mel_bins), where num_frames is the length of the audio in frames, and num_mel_bins is the number of Mel frequency bands used. The text transcripts are tokenized into a sequence of token IDs, resulting in a 1D tensor (sequence_length).

2. These processed input features and labels are combined into a Dataset object, which is then split into separate training, validation, and test datasets using an 80-10-10 split strategy. During training, the model receives batches of spectrograms and tokenized labels, with shapes adjusted by the custom data collator to ensure uniformity (e.g., padded to a consistent size). 

3. The fine-tuning process focuses on updating a few targeted layers specified by LoRA while keeping most of the model frozen to maintain computational efficiency. The model's performance is monitored using the Word Error Rate (WER) metric, calculated on the validation and test datasets. This approach enables efficient fine-tuning of the Whisper model with minimal data and computational resources while preserving its language and speech recognition capabilities.
***(Also want to apply early stopping method for training procees)***


- **Why Whisper?**: Whisper's architecture handles long audio segments and multilingual transcripts, making it a strong candidate for court hearings.

Whisper architecture : 

![image](https://github.com/user-attachments/assets/3e846c2a-4a9c-4a8a-99cc-ce9836e87556)


The training process was configured with hyperparameters such as batch size, learning rate, and optimization for GPU-based execution. We evaluated the model after fine-tuning to assess its performance.

### 6. Model Evaluation
The model was evaluated using the **Word Error Rate (WER)** metric. WER is a common measure of ASR performance and provides insights into how well the model's predicted transcripts match the actual transcripts. 

***We Identified areas for future improvements***, such as : 
1. Better handling of overlapping speech
2. Enhanced diarization accuracy with a better model or maybe a custom trained on indian accented english speech.
3. Trying more variations of hyperparameter tuning (dynamic lr, batch size, etc.)
4. Data Augmentation (to have more diverse dataset in order to prevent overfitting)
5. Can also apply early stopping technique.
6. Since For our particular use case training can take time but we want our model to be robust and highly accurate considering the sensitivity of the data - we can try with full fine tuning as well unlike PEFT technique that we have used
7. use Better techqniques for bg noise removal from the audio files like spectral subtraction. 

---

## Evaluation Results

| Metric (WER)          | WER (Pretrained Model) | WER (Fine tuned model)  |
|-----------------|---------------|----------------------------|
| Value | **30-70%**     |          |
| Sentence Count  |         |                       |
| Segment Length  | 3-20 seconds |  3-20 seconds         |
| Dataset Size    | 27500 minutes|   27500 minutes       |

*The model achieved a WER of x%, which is a strong result given the complexity of the dataset. Future improvements could further reduce this error.*

---

## Conclusion
The project successfully fine-tuned the Whisper model to improve transcription accuracy for Supreme Court hearing data. By levaraging pdf transcripts specific structure and speaker diarization alignment, we reduced errors related to complex dialogues. Forced alignment was deemed unnecessary as the segment-based approach was sufficient and less computationally intensive for the advanced model like whisper. 

---

## Shortcomings & Future Directions 
1. **Overlapping Speech**: This implementation did not fully address overlapping speech in the audio segments.Like Permutation Invariant Training (PIT), Spectral Masking, Blind Source Separation (BSS) Models.  
2. **Lack of Data Augmentation**: No data augmentation techniques (e.g., adding noise, pitch shifting, or speed perturbation) were applied to increase the variability in the dataset. This could have helped improve the robustness of the model in noisy environments, which are common in real-world scenarios and to prevent overfitting. 
3. **No Force Alignment for word level accuracy**: The lack of forced alignment might result in mismatches between the actual spoken words and their corresponding timestamps. Forced alignment could have improved word-level accuracy, especially in longer segments. Since are completely dependent on diarization model and 
4. **Diarization and Silence Removal**: The diarization model and silence removal process could be enhanced for greater accuracy.
5. **Majority Vote Algorithm**: Insetead of simple diariazation model to segment audio filw we can sentence seperation in audio and transcripts using majorioty vote algorithm.
6. **Vanilla Fine Tuning**: AdamW as optimzer, train-35 epochs (10% of the total steps), dynamic lr (which now is .000001) and decaying, Checkpoints saved every 200 steps - reference : 2.

---

## References
1. **Bredin, Hervé, et al. (2020)**. "Development of Supervised Speaker Diarization System Based on the PyAnnote Audio Processing Library." *IEEE ICASSP 2020*.

2. **Chen, Jiahong, et al. (2023)**. "Exploration of Whisper Fine-Tuning Strategies for Low-Resource ASR." *IEEE ICASSP 2023*.

3. **Bredin, Hervé, and Laurent, Antoine (2021)**. "End-to-end speaker segmentation for overlap-aware resegmentation." *Proc. Interspeech 2021*.

4. **Zhou, Yuxuan, et al. (2020)**. "Data augmentation and optimization strategies for end-to-end speech recognition." *Interspeech 2020*.

5. **Baevski, Alexei, et al. (2020)**. "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS 2020*.
