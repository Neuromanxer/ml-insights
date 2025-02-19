from preprocessing import preprocess_data
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from preprocessing import preprocess_data

def label_clusters_general(df, cluster_col='Cluster', feature_1='income', feature_2='purchase_frequency'):
    """ Assigns meaningful labels to customer segments based on two key features dynamically. """

    # Determine thresholds using quantiles (more flexible than median)
    q1 = df[feature_1].median()
    q2 = df[feature_2].median()

    labels = {
        (True, True): f"High-{feature_1}, High-{feature_2}",
        (True, False): f"High-{feature_1}, Low-{feature_2}",
        (False, True): f"Low-{feature_1}, High-{feature_2}",
        (False, False): f"Low-{feature_1}, Low-{feature_2}",
    }

    df["Segment"] = df.apply(lambda row: labels[(row[feature_1] > q1, row[feature_2] > q2)], axis=1)
    return df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_customer_segments(df, feature_1='income', feature_2='purchase_frequency'):
    """ Generalized function to visualize customer segments based on two features. """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature_1, y=feature_2, hue="Segment", palette="viridis", s=100, alpha=0.7)
    
    plt.xlabel(feature_1.replace("_", " ").title())  # Clean column names for better readability
    plt.ylabel(feature_2.replace("_", " ").title())
    plt.title("Customer Segmentation Based on Key Features")
    plt.legend(title="Segment")
    plt.show()

def find_optimal_k(data, max_k=10):
    """ Finds the best K using Elbow & Silhouette method """
    wcss = []
    silhouette_scores = []

    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    best_k = np.argmax(silhouette_scores) + 2  # Silhouette optimal K
    return best_k, wcss, silhouette_scores

def run_kmeans(df, drop_columns=[]):
    """ Runs KMeans clustering on dataset """
    data = preprocess_data(df, drop_columns)

    best_k, wcss, silhouette_scores = find_optimal_k(data)
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    
    df['Cluster'] = clusters
    return df, kmeans, best_k