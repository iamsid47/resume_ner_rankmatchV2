import os
import PyPDF2
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pdfplumber import open as pdf_open
from calculate_similarity import calculate_tfidf_similarity, calculate_sentence_transformer_similarity, calculate_levenshtein_similarity, calculate_bert_similarity as calculate_euclidean_similarity, calculate_spacy_similarity, calculate_simhash_similarity
from flask_cors import CORS
import time
from calculate_similarity import extract_ner_from_cvs

app = Flask(__name__)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            text += page_text
        return text

def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

@app.route('/process_data', methods=['POST'])
def process_data():
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    try:
        if 'job_description' not in request.files or 'resume_files' not in request.files:
            return jsonify({'error': 'Missing job description or resume files'}), 400
        
        job_description = request.files['job_description']
        resume_files = request.files.getlist('resume_files')
        
        job_description_text = ""
        with pdf_open(job_description) as pdf:
            for page in pdf.pages:
                try:
                    job_description_text += page.extract_text()
                except UnicodeDecodeError:
                    job_description_text += page.extract_text(encoding='latin-1')
        
        output_folder = 'output_txts'
        os.makedirs(output_folder, exist_ok=True)

        print("Input files done")
        
       
        cv_texts = []
        result = []
        
        for resume_file in resume_files:
            if resume_file.filename == '':
                continue
            
            resume_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(output_folder, resume_filename)
            txt_filename = os.path.splitext(resume_filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)
            
            resume_file.save(resume_path)
            
            resume_text = pdf_to_text(resume_path)
            cv_texts.append(resume_text)
            
            preprocessed_resume_text = preprocess_text(resume_text)
            
            tfidf_similarity = calculate_tfidf_similarity(job_description_text, preprocessed_resume_text)
            sentence_transformer_similarity = calculate_sentence_transformer_similarity(job_description_text, preprocessed_resume_text)
            levenshtein_similarity = calculate_levenshtein_similarity(job_description_text, preprocessed_resume_text)
            euclidean_similarity = calculate_euclidean_similarity(job_description_text, preprocessed_resume_text)
            simhash_similarity = calculate_simhash_similarity(job_description_text, preprocessed_resume_text)
            spacy_similarity = calculate_spacy_similarity(job_description_text, preprocessed_resume_text, 'en_core_web_lg')
            
  

            # weight_tfidf = 0.0425
            # weight_sentence_transformer = 0.0350
            # weight_levenshtein = 0.0350
            # weight_euclidean = 0.0375
            # weight_spacy = 0.7
            # weight_simhash = 0.15

            weight_tfidf = 0.07
            weight_sentence_transformer = 0.07
            weight_levenshtein = 0.045
            weight_euclidean = 0.01
            weight_spacy = 0.64
            weight_simhash = 0.175

            ensemble_score = (
                (tfidf_similarity * weight_tfidf)
                + (sentence_transformer_similarity * weight_sentence_transformer)
                + (levenshtein_similarity * weight_levenshtein)
                # + (euclidean_similarity * weight_euclidean)
                + (spacy_similarity * weight_spacy)
                + (simhash_similarity * weight_simhash)
            ) / (weight_tfidf + weight_sentence_transformer + weight_levenshtein  + weight_spacy + weight_simhash)

            print("FILE NAME: ", resume_filename)
            print("TF-IDF Similarity:", tfidf_similarity)
            print("Sentence Transformer Similarity:", sentence_transformer_similarity)
            print("Levenshtein Similarity:", levenshtein_similarity)
            print("Euclidean Similarity:", euclidean_similarity)
            print("Spacy Similarity:", spacy_similarity)
            print("Simhash Similarity:", simhash_similarity)

            print("Ensemble Score:", ensemble_score)
            rounded_score = float(f"{ensemble_score:.4f}")
            ranked_score = rounded_score * 100

            result.append({
                'resume_filename': resume_filename,
                'ensemble_score': ranked_score,
                

            })
        
        ner_results = extract_ner_from_cvs(cv_texts)

        result.sort(key=lambda x: x['ensemble_score'], reverse=True)

        
        ranked_result = []
        for rank, (data, ner_result) in enumerate(zip(result, ner_results), start=1):
            ranked_result.append({
                'rank': rank,
                'resume_filename': data['resume_filename'],
                'ensemble_score': data['ensemble_score'],
                'ner_output': ner_result
            })


        end_time = time.time()

        resp_time = end_time - start_time
        print("Response Time: ", resp_time)
        return jsonify({'result': ranked_result}), 200
        
    
    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
