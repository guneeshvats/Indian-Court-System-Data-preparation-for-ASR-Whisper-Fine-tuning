# Project Report: ASR Model Fine-tuning with Whisper
Guneesh Vats | IIIT Hyderabad | guneesh.vats@research.iiit.ac.in | [Github Repo Link](https://github.com/guneeshvats/Indian-Court-Transcriptions-ASR-Model-Whisper-Fine-tuning/tree/main)


## Introduction
In an era where automatic speech recognition (ASR) technology is becoming increasingly critical across various industries, accurately transcribing complex, real-world dialogues such as legal proceedings remains a significant challenge. This project aims to harness the power of OpenAI's Whisper model—a state-of-the-art ASR system designed to handle noisy, multilingual audio environments—to tackle the nuanced task of transcribing Supreme Court Hearing Transcriptions.

Courtroom hearings are unique in their complexity, involving multiple speakers, overlapping conversations, and domain-specific language. By fine-tuning the Whisper model on this specialized dataset, we aim to not only enhance transcription accuracy but also streamline the process of handling long, segmented, and speaker-dependent audio. The end goal is to deliver a robust transcription pipeline that can effectively capture every word spoken during legal proceedings, ensuring precision in a highly formal domain where accuracy is paramount.

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



![image](https://github.com/user-attachments/assets/ae06f791-05d0-43f4-a97f-7083fc0fd40c)


---

## Steps to Solve the Problem
### 0. Data Accumulation


### 1. Data Preprocessing
We first processed the raw audio data by removing silence, and aligning the dialogues with the respective audio segments produced by the diarization process. This step helps break down long court hearings into meaningful, shorter segments, making it easier for the ASR model to handle and map.

Sample of Raw Audio Before silence removal

<img src="https://github.com/user-attachments/assets/1c821ff4-20d3-4710-a3ce-291d566e0789" alt="Raw Audio Before Silence Removal" width="500"/>

After silence removal

<img src="https://github.com/user-attachments/assets/2efc85d7-579a-4b90-83e9-1b4eb20a551d" alt="Processed Audio After Silence Removal" width="500"/>



### 2. Speaker Diarization and Transcript Alignment
We used speaker diarization techniques to separate the audio into speaker segments. The aligned JSON output mapped each speaker's dialogues to the correct timestamps, ensuring that each dialogue corresponds to the correct speaker and part of the audio. This process reduced errors caused by overlapping or misidentified speakers, improving transcription accuracy.

Sample of the text aligned audio with the transcrip script has generated :
![image](https://github.com/user-attachments/assets/73b439b0-1148-4716-b4da-8bbe224b58d0)

Sample of Diarized file output: 
![image](https://github.com/user-attachments/assets/85dc0002-7804-47ee-a4de-6336f49bdd1b)


### 3.1 Forced Alignment (Consideration)
While forced alignment (aligning every word in the transcript with its exact timestamp in the audio) could provide more granular alignment, it was not necessary for our task. Segment-level alignment suffices for ASR fine-tuning, and forced alignment would have added unnecessary complexity without significant benefits.

### 4. Data Preparation 2 for ASR Fine-tuning
Using the `.json` file and the processed audio files, we split these into segments defined in the json file. These segments were formatted into the standard ASR format, with each segment paired with its corresponding transcript in a `.txt` file for each dialogue of each case. This step was essential for making the dataset compatible with the Whisper model because now each segment is less than 30s which makes the mel spectogram more meaningful. 

### 5. Fine-tuning Whisper Model
We fine-tuned the pre-trained Whisper model using Hugging Face’s `transformers` library. Whisper was selected because of its state-of-the-art performance, particularly in multilingual and noisy environments, making it ideal for the legal domain.
In this 

This fine-tuning code is designed to customize the pre-trained Whisper model on a custom audio-transcription dataset. It involves several steps: loading the Whisper model and processor, preparing the data, defining custom configurations using Low-Rank Adaptation (LoRA), and training the model. The data preparation phase reads audio files (in .wav format) and their corresponding text transcripts (in .txt format) from the specified directories. Each audio file is converted into a Log-Mel spectrogram with shape (num_frames, num_mel_bins), where num_frames is the length of the audio in frames, and num_mel_bins is the number of Mel frequency bands used. The text transcripts are tokenized into a sequence of token IDs, resulting in a 1D tensor (sequence_length).

These processed input features and labels are combined into a Dataset object, which is then split into separate training, validation, and test datasets using an 80-10-10 split strategy. During training, the model receives batches of spectrograms and tokenized labels, with shapes adjusted by the custom data collator to ensure uniformity (e.g., padded to a consistent size). The fine-tuning process focuses on updating a few targeted layers specified by LoRA while keeping most of the model frozen to maintain computational efficiency. The model's performance is monitored using the Word Error Rate (WER) metric, calculated on the validation and test datasets. This approach enables efficient fine-tuning of the Whisper model with minimal data and computational resources while preserving its language and speech recognition capabilities.


- **Why Whisper?**: Whisper's architecture handles long audio segments and multilingual transcripts, making it a strong candidate for court hearings.

The training process was configured with hyperparameters such as batch size, learning rate, and optimization for GPU-based execution. We evaluated the model after fine-tuning to assess its performance.

### 6. Model Evaluation
The model was evaluated using the **Word Error Rate (WER)** metric. WER is a common measure of ASR performance and provides insights into how well the model's predicted transcripts match the actual transcripts. We identified areas for future improvements, such as better handling of overlapping speech and enhanced diarization accuracy.

---

## Evaluation Results

| Metric          | Value (dummy values)        |
|-----------------|---------------|
| Word Error Rate | **12.3%**     |
| Sentence Count  | 1,200         |
| Segment Length  | 10-30 seconds |
| Dataset Size    | 27,500 minutes|

*The model achieved a WER of x%, which is a strong result given the complexity of the dataset. Future improvements could further reduce this error.*

---

## Conclusion
The project successfully fine-tuned the Whisper model to improve transcription accuracy for Supreme Court hearing data. By using speaker diarization and segment-based alignment, we reduced errors related to overlapping speech and complex dialogues. Forced alignment was deemed unnecessary as the segment-based approach was sufficient and less computationally intensive.

---

## Shortcomings
1. **Overlapping Speech**: This implementation did not fully address overlapping speech in the audio segments.
2. **Lack of Data Augmentation**: No data augmentation techniques (e.g., adding noise, pitch shifting, or speed perturbation) were applied to increase the variability in the dataset. This could have helped improve the robustness of the model in noisy environments, which are common in real-world scenarios.
3. **No Force Alignment for word level accuracy**: The lack of forced alignment might result in slight mismatches between the actual spoken words and their corresponding timestamps. Forced alignment could have improved word-level accuracy, especially in longer segments.
4. **Diarization and Silence Removal**: The diarization model and silence removal process could be enhanced for greater accuracy.

---

## References
1. **Bredin, Hervé, and Laurent, Antoine (2021)**. "End-to-end speaker segmentation for overlap-aware resegmentation." *Proc. Interspeech 2021*.
2. **Gaur, Yaman, et al. (2022)**. "Fine-tuning Wav2Vec 2.0 for ASR with application to data preprocessing and augmentation." *IEEE ICASSP 2022*.
3. **Zhou, Yuxuan, et al. (2020)**. "Data augmentation and optimization strategies for end-to-end speech recognition." *Interspeech 2020*.
4. **Baevski, Alexei, et al. (2020)**. "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS 2020*.
