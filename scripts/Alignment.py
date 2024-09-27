############################################################################################################################
############################################################################################################################
'''
Purpose : Extract dialogues from PDF transcripts, align them with speaker diarization results from RTTM files, and create a 
structured JSON output for each case.

Steps : 

1. Extract Dialogues: Reads dialogues from PDF transcripts while ignoring metadata and timestamps.
2. Parse RTTM Files: Extracts speaker information, start time, and duration from diarization results.
3. Align Dialogues: Maps PDF dialogues to diarized speaker segments and saves the aligned data in JSON format.
4. Process All Files: Iterates through all RTTM and PDF files in the specified folders.

Preprocessing Steps :
1. Expand Contractions: Replaces contractions like "don't" with "do not" using `expand_contractions`.
2. Remove Filler Words: Removes common filler words like "um" and "you know" from the dialogue using `remove_filler_words`.
3. Spell Out Numbers: Converts numerical digits to their word equivalents using `spell_out_numbers`.
4. Clean Speaker Names and Dialogue: Removes digits and unwanted characters from speaker names and dialogues.


Replace Paths : 

rttm_folder = "path_to_rttm_files"
pdf_folder = "path_to_pdf_files"
output_json_folder = "path_to_save_output_json"

Code written by: Guneesh Vats
Date: 25th Sept, 2024
'''
############################################################################################################################
############################################################################################################################


import pdfplumber
import os
import json
import re
import inflect

# Initialize the inflect engine for number-to-word conversion
inflect_engine = inflect.engine()

# Dictionary for expanding contractions
contractions_dict = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "I'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "there's": "there is",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
}

# List of filler words to remove
filler_words = ["um", "uh", "you know"]

# Function to expand contractions
def expand_contractions(text, contractions_dict):
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

# Function to remove filler words
def remove_filler_words(text, filler_words):
    pattern = re.compile(r'\b(' + '|'.join(filler_words) + r')\b', re.IGNORECASE)
    return pattern.sub('', text).strip()

# Function to spell out numbers
def spell_out_numbers(text):
    return re.sub(r'\b\d+\b', lambda x: inflect_engine.number_to_words(x.group()), text)

# Preprocess function that combines all the preprocessing steps
def preprocess_text(text):
    text = expand_contractions(text, contractions_dict)  # Expand contractions
    text = remove_filler_words(text, filler_words)        # Remove filler words
    text = spell_out_numbers(text)                        # Convert numbers to words
    return text

# Modified function to clean dialogue text with added preprocessing steps
def clean_dialogue_text(dialogue):
    dialogue = re.sub(r'\d+$', '', dialogue).strip()  # Remove digits at the end
    dialogue = preprocess_text(dialogue)              # Apply preprocessing
    return dialogue

# Step 1: Helper function to check if a line is a valid speaker line (not a line number or timestamp)
def is_valid_speaker_line(line):
    line = line.strip()
    return not line.isdigit() and ":" in line

# Step 2: Helper function to check if a line is a timestamp or irrelevant metadata
def is_timestamp_or_metadata(line):
    timestamp_pattern = r"\b\d{1,2}:\d{2}(:\d{2})?\b"
    return re.match(timestamp_pattern, line.strip()) is not None

# Step 3: Helper function to clean real speaker names by removing any digits
def clean_speaker_name(speaker):
    return re.sub(r'\d+', '', speaker).strip()

# Step 4: Helper function to remove trailing numbers from dialogue text
def clean_dialogue_text(dialogue):
    # Remove any digits at the end of the dialogue
    return re.sub(r'\d+$', '', dialogue).strip()

# Step 5: Extract dialogues from the PDF, ignoring timestamps and line numbers, and preprocess them
def extract_dialogues_from_pdf(pdf_path):
    dialogues = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:  # Skip the first page and process from the second page onwards
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                current_speaker = None
                current_dialogue = []

                for line in lines:
                    if is_timestamp_or_metadata(line):
                        continue

                    if is_valid_speaker_line(line):  
                        if current_speaker is not None:
                            full_dialogue = " ".join(current_dialogue).strip()
                            dialogues.append({
                                "speaker": current_speaker,
                                "dialogue": clean_dialogue_text(full_dialogue)
                            })

                        speaker, dialogue = line.split(":", 1)
                        current_speaker = clean_speaker_name(speaker.strip())  
                        current_dialogue = [dialogue.strip()]  
                    else:
                        current_dialogue.append(line.strip())

                if current_speaker and current_dialogue:
                    full_dialogue = " ".join(current_dialogue).strip()
                    dialogues.append({
                        "speaker": current_speaker,
                        "dialogue": clean_dialogue_text(full_dialogue)
                    })
    return dialogues
# Step 6: Parse the RTTM diarization file
def parse_rttm(rttm_path):
    speaker_segments = []
    with open(rttm_path, "r") as file:
        for line in file:
            parts = line.split()
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            speaker_segments.append({
                "start_time": start_time,
                "end_time": start_time + duration,
                "speaker": speaker  # "Speaker 0", "Speaker 1", etc.
            })
    return speaker_segments

# Step 7: Align the extracted dialogues with diarization and map speaker names
def align_dialogues_with_diarization(diarization_path, pdf_path, output_json_path):
    dialogues = extract_dialogues_from_pdf(pdf_path)
    speaker_segments = parse_rttm(diarization_path)

    result = []
    dialogue_index = 0  # Track which dialogue we are aligning

    # Assuming that the diarization order matches the order in the PDF
    speaker_mapping = {}  # To map diarized speakers to real names

    for segment in speaker_segments:
        diarized_speaker = segment["speaker"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]

        # Align the next available dialogue with the current diarization segment
        if dialogue_index < len(dialogues):
            dialogue = dialogues[dialogue_index]
            real_speaker = dialogue["speaker"]

            # Clean the real speaker name by removing any digits
            clean_real_speaker = clean_speaker_name(real_speaker)

            # If we haven't yet mapped this diarized speaker, assign the cleaned real name
            if diarized_speaker not in speaker_mapping:
                speaker_mapping[diarized_speaker] = clean_real_speaker

            result.append({
                "diarized_speaker": diarized_speaker,  # "Speaker 0", "Speaker 1", etc.
                "real_speaker": speaker_mapping[diarized_speaker],  # Cleaned real speaker name from PDF
                "transcript": dialogue["dialogue"],  # Extracted dialogue from PDF
                "start_time": start_time,
                "end_time": end_time
            })
            dialogue_index += 1  # Move to the next dialogue

    # Save the aligned data to JSON
    with open(output_json_path, "w") as output_file:
        json.dump(result, output_file, indent=4)
    print(f"Alignment completed and saved to: {output_json_path}")

# Step 8: Process all RTTM and PDF files in the folder
def process_all_files(rttm_folder, pdf_folder, output_json_folder):
    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)

    # Iterate through all the RTTM files in the folder
    for rttm_file in os.listdir(rttm_folder):
        if rttm_file.endswith(".rttm"):
            case_name = rttm_file.replace(".rttm", "")  # Extract the base name (e.g., 'case_1')
            rttm_path = os.path.join(rttm_folder, rttm_file)
            pdf_path = os.path.join(pdf_folder, case_name + ".pdf")

            # Check if the corresponding PDF exists
            if os.path.exists(pdf_path):
                output_json_path = os.path.join(output_json_folder, case_name + ".json")
                print(f"Processing {case_name}...")
                align_dialogues_with_diarization(rttm_path, pdf_path, output_json_path)
            else:
                print(f"Warning: PDF for {case_name} not found!")

rttm_folder = 'output_diarization_folder'  # Folder containing RTTM files
pdf_folder = 'pdf_transcripts'    # Folder containing PDF transcripts
output_json_folder = 'final_output_json'  # Folder where aligned JSONs will be saved

process_all_files(rttm_folder, pdf_folder, output_json_folder)
