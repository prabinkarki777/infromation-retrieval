import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('punkt')

# Load data


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Configure TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')

# Initialize stopwords and Porter Stemmer
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed_tokens = [porter.stem(token)
                      for token in tokens if token not in stop_words]
    return ' '.join(stemmed_tokens)


# Combine text fields
def combine_fields(publication):
    title = preprocess_text(publication['title'])
    authors = ' '.join([preprocess_text(author['name'])
                       for author in publication['authors']])
    # Combine title and authors
    combined_text = f"{title} {authors}"
    return combined_text


def perform_search(query):
    data = load_data('search_engine/research_output.json')

    # Preprocess documents
    processed_documents = [combine_fields(publication) for publication in data]

    # Fit TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform(processed_documents)

    # Process the query
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(
        query_vector, tfidf_matrix).flatten()
    indices_sorted_by_relevance = cosine_similarities.argsort()[::-1]

    # Prepare results
    results = []
    for idx in indices_sorted_by_relevance:
        relevance_score = cosine_similarities[idx]
        if relevance_score > 0.02:
            authors = [{'name': author['name'], 'profile_link': author.get('profile_link', '')}
                       for author in data[idx]['authors']]
            result = {
                'title': data[idx]['title'],
                'authors': authors,
                'year': data[idx]['publication_year'],
                'link': data[idx]['publication_link'],
                'relevance_score': round(relevance_score, 2)
            }
            results.append(result)

    return results
