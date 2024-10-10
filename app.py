from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups #20 newsgroups dataset from Kaggle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk #natural language tool kit
from nltk.corpus import stopwords #common words like "the", "and" etc.

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
#Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all') #download 20newsgroups dataset
documents = newsgroups.data #extracts text data from dataset into a list called documents

#Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english') #remove common english words and convert documents into a matrix of TF-IDF features
X_tfidf = vectorizer.fit_transform(documents)

# Apply SVD to reduce dimensionality
lsa = TruncatedSVD(n_components=100, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = vectorizer.transform([query])
    
    # Project the query into the LSA space
    query_lsa = lsa.transform(query_tfidf)
    
    # Compute cosine similarities between the query and all documents
    similarities = cosine_similarity(query_lsa, X_lsa)[0]
    
    # Get the top 5 most similar documents
    top5_indices = similarities.argsort()[-5:][::-1]
    top5_documents = [documents[i] for i in top5_indices]
    top5_similarities = similarities[top5_indices]
    
    return top5_documents, top5_similarities.tolist(), top5_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(host='localhost', port=3000)
