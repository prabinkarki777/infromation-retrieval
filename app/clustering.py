import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import csv
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Ensure Matplotlib is in non-GUI mode
matplotlib.use('Agg')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class DocumentClustering:
    def __init__(self, file_path='search_engine/ny-times-news-data.csv'):
        self.file_path = file_path
        self.df = None

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=3
        )
        self.kmeans = None
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
            return False
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
            return False
        return True

    def preprocess_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(
                token) for token in tokens if token.isalpha() and token not in self.stopwords]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocessing text: {e}")
            return ""

    def train_model(self):
        if self.df is None and not self.load_data():
            return
        if 'description' not in self.df.columns:
            print("Column 'description' not found in the dataset.")
            return

        # Preprocess each document in the 'description' column
        self.df['processed_text'] = self.df['description'].apply(
            self.preprocess_text)

        # Convert the preprocessed text into a list
        documents = self.df['processed_text'].dropna().tolist()
        if not documents:
            print("No documents to cluster.")
            return

        # Vectorize the documents
        X = self.vectorizer.fit_transform(documents)
        print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")

        # Cluster the documents
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.kmeans.fit(X)

        # Get cluster labels and counts
        labels = self.kmeans.labels_

        pca = PCA(n_components=2)
        x_reduce = pca.fit_transform(X.toarray())
        silhouette_Score = silhouette_score(x_reduce, labels)
        print("Silhouette Score:", silhouette_Score)

        # Get cluster labels and counts
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print("\nCluster Counts:")
        print(cluster_counts)

        # Print top terms per cluster
        self.print_top_terms_per_cluster(X)
        # self.plot_top_terms_per_cluster(X)
        self.plot_clusters(X, labels, x_reduce)

    def print_top_terms_per_cluster(self, X, top_n=10):
        feature_names = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        for i, centroid in enumerate(order_centroids):
            print(f"Cluster {i} Top Terms:")
            terms = [feature_names[ind]
                     for ind in centroid[:top_n] if ind < len(feature_names)]
            print(", ".join(terms))
            print()

    def plot_top_terms_per_cluster(self, X, top_n=10):
        feature_names = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        for i, centroid in enumerate(order_centroids):
            terms = [feature_names[ind]
                     for ind in centroid[:top_n] if ind < len(feature_names)]
            term_weights = [self.kmeans.cluster_centers_[i][ind]
                            for ind in centroid[:top_n] if ind < len(feature_names)]

            axes[i].barh(terms, term_weights, color='skyblue')
            # Invert y-axis to display the highest weight on top
            axes[i].invert_yaxis()
            axes[i].set_xlabel('Weight')
            axes[i].set_title(f'Top {top_n} Terms for Cluster {i}')

        plt.tight_layout()
        plt.savefig('clusters_top_terms.png')
        plt.close()

    def plot_clusters(self, X, labels, x_reduce):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            x_reduce[:, 0], x_reduce[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title('Document Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig('document_clusters.png')
        plt.close()

    def predict_cluster(self, new_document):
        if self.kmeans is None:
            self.train_model()
        if self.kmeans is not None:
            processed_doc = self.preprocess_text(new_document)
            if processed_doc:
                new_X = self.vectorizer.transform([processed_doc])
                predicted_cluster = self.kmeans.predict(new_X)[0]
                return predicted_cluster
        return None

    def cluster_document(self, new_document):
        predicted_cluster = self.predict_cluster(new_document)
        cluster_names = {0: "entertainment", 1: "politics", 2: "economy"}

        if predicted_cluster is not None:
            cluster_name = cluster_names.get(predicted_cluster, 'unknown')
            response = {
                'document': new_document,
                'predicted_cluster': cluster_name
            }
            return response
        else:
            return {'error': 'Failed to predict cluster.'}, 500

    def save_new_document(self, new_document, category):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([category.lower(), new_document])
        print(f"New document saved with category '{category}'.")
