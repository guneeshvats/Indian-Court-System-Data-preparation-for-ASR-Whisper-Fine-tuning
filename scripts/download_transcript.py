import os
import requests
import pandas as pd

# Directory to save the downloaded PDFs
pdf_directory = 'court_transcripts_pdfs'
os.makedirs(pdf_directory, exist_ok=True)

# Load the dataset (replace with the correct path to your dataset)
file_path = 'SC Transcripts ML Engineer Assignment - ML Engineer Assignment.csv'
data = pd.read_csv(file_path)

# Loop through the dataset and download each PDF
for index, row in data.iterrows():
    case_name = row['Case Name']  # Adjust the column name if necessary
    pdf_link = row['Transcript Link']  # Adjust the column name if necessary

    if pd.isna(pdf_link):
        print(f"Skipping download for row {index} due to missing mp3 link.")
        continue

    # Check if case_name is NaN or not
    if pd.isna(case_name):
        case_name = f"case_{index}"  # Use a default name with the index if case_name is NaN
    
    # Create a safe file name
    safe_case_name = case_name.replace(' ', '_').replace('/', '_').replace(':', '_')
    pdf_file_name = f"{safe_case_name}.pdf"
    pdf_file_path = os.path.join(pdf_directory, pdf_file_name)
    
    # Download the PDF
    try:
        response = requests.get(pdf_link)
        response.raise_for_status()  # Check for HTTP errors
        with open(pdf_file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {pdf_file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdf_file_name}: {e}")

print("All PDFs have been downloaded.")
