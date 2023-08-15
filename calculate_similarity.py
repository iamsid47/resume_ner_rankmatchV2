# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer, util
# from nltk.metrics.distance import edit_distance
# from scipy.spatial.distance import euclidean
# import spacy
# from simhash import Simhash



# def calculate_tfidf_similarity(text1, text2):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
#     similarity_score = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    
#     return similarity_score

# def calculate_sentence_transformer_similarity(text1, text2):
#     model = SentenceTransformer('LaBSE')
#     embedding1 = model.encode([text1], convert_to_tensor=True)
#     embedding2 = model.encode([text2], convert_to_tensor=True)
#     similarity_score = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
#     return similarity_score

# def calculate_levenshtein_similarity(text1, text2):
#     levenshtein_score = 1 - edit_distance(text1, text2) / max(len(text1), len(text2))
#     return levenshtein_score

# def calculate_bert_similarity(text1, text2):
#     model = SentenceTransformer('LaBSE')
#     embedding1 = model.encode([text1], convert_to_tensor=True)[0]
#     embedding2 = model.encode([text2], convert_to_tensor=True)[0]
#     similarity_score = 1 / (1 + euclidean(embedding1, embedding2))
#     return similarity_score

# def calculate_spacy_similarity(text1, text2, model_name):
#     nlp = spacy.load(model_name)
#     doc1 = nlp(text1)
#     doc2 = nlp(text2)
#     similarity_score = doc1.similarity(doc2)
#     return similarity_score

# def calculate_simhash_similarity(text1, text2):
#     hash1 = Simhash(text1)
#     hash2 = Simhash(text2)
#     similarity = 1 - (hash1.distance(hash2) / 64)  # Normalize the distance to similarity
#     return similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import euclidean
import spacy
from simhash import Simhash
import torch

def calculate_tfidf_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity_score = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    
    return similarity_score

def calculate_sentence_transformer_similarity(text1, text2):
    model = SentenceTransformer('LaBSE')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embedding1 = model.encode([text1], convert_to_tensor=True).to(device)
    embedding2 = model.encode([text2], convert_to_tensor=True).to(device)
    
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    return similarity_score

def calculate_levenshtein_similarity(text1, text2):
    levenshtein_score = 1 - edit_distance(text1, text2) / max(len(text1), len(text2))
    return levenshtein_score

def calculate_bert_similarity(text1, text2):
    model = SentenceTransformer('LaBSE')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embedding1 = model.encode([text1], convert_to_tensor=True).to(device)[0]
    embedding2 = model.encode([text2], convert_to_tensor=True).to(device)[0]
    
    similarity_score = 1 / (1 + euclidean(embedding1.cpu(), embedding2.cpu()))
    return similarity_score

def calculate_spacy_similarity(text1, text2, model_name):
    nlp = spacy.load(model_name)
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

def calculate_simhash_similarity(text1, text2):
    hash1 = Simhash(text1)
    hash2 = Simhash(text2)
    similarity = 1 - (hash1.distance(hash2) / 64)  # Normalize the distance to similarity
    return similarity




