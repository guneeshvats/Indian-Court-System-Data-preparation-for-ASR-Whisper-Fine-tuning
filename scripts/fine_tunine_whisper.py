from datasets import load_dataset, DatasetDict
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_metric

# Load your dataset
def load_audio(batch):
    speech_array, _ = librosa.load(batch["audio"], sr=16000)
    batch["speech"] = speech_array
    return batch

# Assuming your dataset is a CSV or JSON
dataset = load_dataset("csv", data_files="/path_to_your_dataset.csv")

# Process the audio and load it into the dataset
dataset = dataset.map(load_audio)



# Load Whisper model and tokenizer
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Preprocess the Data
def preprocess_data(batch):
    # Process audio
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt").input_features
    # Tokenize transcripts
    with processor.as_target_processor():
        labels = processor(batch["transcript"]).input_ids

    batch["input_features"] = inputs.squeeze().detach().cpu().numpy()
    batch["labels"] = labels
    return batch

# Apply preprocessing
processed_dataset = dataset.map(preprocess_data, remove_columns=["speech", "transcript", "audio"])


# Fine tuning the configuration 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/path_to_output_model/",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # If using a GPU with half-precision support
)

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor.feature_extractor,
)

# Fine-tune Whisper model
trainer.train()


# Evaluating the model

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids = pred.label_ids
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# Run evaluation
results = trainer.evaluate()
print(results)
