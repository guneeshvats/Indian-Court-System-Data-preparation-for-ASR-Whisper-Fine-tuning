############################################################################################################################
############################################################################################################################
'''
Purpose : Fine tune Whisper model on custom dataset 

Steps : 

1. Load the Model & Processor: Initializes the pre-trained Whisper model and its processor for handling audio and text.
2. Data Preparation: Reads audio and transcript files from a local directory, processes audio into Log-Mel spectrograms, 
   and converts transcripts into token IDs.
3. Apply LoRA Fine-Tuning: Customizes the model with LoRA parameters for resource-efficient fine-tuning.
4. Training & Evaluation: Fine-tunes the model on provided data and evaluates using Word Error Rate (WER).
5. Checkpointing: Saves model checkpoints during training for future reference.

Replace Paths : 

audio_folder = "path_to_your_audio_folder"
transcript_folder = "path_to_your_transcripts_folder"


Code written by: Guneesh Vats
Date: 25th Sept, 2024
'''
############################################################################################################################
############################################################################################################################


from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftModelForSeq2SeqLM, get_peft_model, LoraConfig, TaskType
import os
import librosa
import torch
from evaluate import load
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

# Custom Data Collator for Whisper to handle spectrograms (input_features)
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]
        input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored tokens in loss
        return {"input_features": input_features_padded, "labels": labels_padded}

# 1. Load Pre-trained Whisper Model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Define LoRA Configuration and Custom WhisperTuner Class
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj"]
)

class WhisperTuner(PeftModelForSeq2SeqLM):
    def forward(self, input_features=None, labels=None, **kwargs):
        outputs = self.base_model(input_features=input_features, labels=labels, **kwargs)
        return outputs

model = WhisperTuner(model, lora_config)

# 3. Load Data and Convert to Dataset Format
audio_folder = "/Users/guneeshvats/Desktop/Adalat AI assignment/Approach_2/Audio_Segments_transcripts_splits"
transcript_folder = "/Users/guneeshvats/Desktop/Adalat AI assignment/Approach_2/Audio_Segments_transcripts_splits"
audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
transcript_files = sorted([f for f in os.listdir(transcript_folder) if f.endswith('.txt')])

data = []
for audio_file, transcript_file in zip(audio_files, transcript_files):
    audio_path = os.path.join(audio_folder, audio_file)
    transcript_path = os.path.join(transcript_folder, transcript_file)
    audio, sr = librosa.load(audio_path, sr=16000)
    with open(transcript_path, 'r') as f:
        transcript = f.read().strip()
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
    labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.squeeze(0)
    data.append({"input_features": input_features.numpy().tolist(), "labels": labels.numpy().tolist()})

train_dataset = Dataset.from_dict({"input_features": [x["input_features"] for x in data], "labels": [x["labels"] for x in data]})
eval_dataset = train_dataset

# 4. Define Data Collator
data_collator = CustomDataCollator(processor=processor)

# Custom Trainer Class for Whisper
class CustomWhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_features = torch.tensor(inputs["input_features"]).float().to(model.device)
        labels = torch.tensor(inputs["labels"]).long().to(model.device)
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# 5. Training Arguments with GPU Optimization and Checkpointing
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_lora_finetuned",
    per_device_train_batch_size=2,  # Reduce batch size if you encounter memory issues
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Accumulate gradients to reduce memory footprint
    learning_rate=5e-5,
    num_train_epochs=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100,  # Save checkpoints every 100 steps
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,  # Keep only last 3 checkpoints
    predict_with_generate=True,
    generation_max_length=225,
    fp16=True,  # Enable mixed precision to save memory and improve performance
    report_to=["tensorboard"],  # Enable reporting to Tensorboard for visualization
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# 6. Load Evaluation Metric
wer_metric = load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 7. Initialize Custom Trainer
trainer = CustomWhisperSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Start Training and Save Checkpoints
trainer.train()

# 9. Evaluate the Model
eval_results = trainer.evaluate(eval_dataset=eval_dataset)

# 10. Print WER
print(f"Word Error Rate (WER): {eval_results['eval_wer']}")
