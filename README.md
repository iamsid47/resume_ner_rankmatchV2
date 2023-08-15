# Resume Scoring API

This is a Flask-based API that scores resumes based on their similarity to a provided job description. The API uses various similarity metrics, including TF-IDF, Sentence Transformer, Levenshtein, Euclidean distance, Spacy, and Simhash to calculate an ensemble score for each resume.

## Requirements

```
Python 3.6 or later
Flask
PyPDF2
nltk
pdfplumber
sentence-transformers
spacy
simhash
pdf2image (optional, for PDF image extraction)
```

## Setup

1. Clone this repository to your local machine.

2. Create a virtual environment and activate it:

```sh
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```sh
pip install -r requirements.txt
```

4. Run the Flask app:

```sh
python app.py
```

The API will start on http://localhost:5000.

## Usage

Send a POST request to http://localhost:5000/process_data with the following parameters:

- job_description: The job description PDF file.
- resume_files[]: An array of resume PDF files.

Configuration
The app.py file contains configuration options for different similarity metrics and their weights. You can adjust these weights and metrics to fine-tune the scoring process.

The output_folder variable defines the directory where temporary files and processed text files will be stored.

Notes
The response time of the API is logged and provided in the JSON response.
