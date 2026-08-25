import os
import joblib
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.config import CLUSTERS_DIR, GENERATE_VISUALIZATION
from src.vectorstore import create_or_get_vectorstore
from src.logging_config import setup_logging

logger = setup_logging(__name__)
CENTROIDS_PATH = os.path.join(CLUSTERS_DIR, "source_centroids.joblib")

def train_clustering():
    """Calculate and save the mean embedding (centroid) for each document source."""
    logger.info("Loading vector database for clustering...")
    vectorstore = create_or_get_vectorstore()

    # Extract data
    collection = vectorstore._collection
    data = collection.get(include=["embeddings", "metadatas"])

    embeddings = data["embeddings"]
    metadatas = data["metadatas"]

    if embeddings is None:
        logger.error("No embeddings found in the vectorstore. Run index first.")
        return
        
    logger.info(f"Loaded {len(embeddings)} embeddings. Grouping by source")

    # Group embeddings
    source_embeddings = {}
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "unknown")
        if source not in source_embeddings:
            source_embeddings[source] = []
        source_embeddings[source].append(embeddings[i])

    # Calculate centroid
    centroids = {}
    for source, emb_list in source_embeddings.items():
        arr = np.array(emb_list)
        centroids[source] = np.mean(arr, axis=0)
        logger.info(f"Source '{source}': Centroid computed from {len(emb_list)} chunks")
  
    # Save models
    os.makedirs(CLUSTERS_DIR, exist_ok=True)
    joblib.dump(centroids, CENTROIDS_PATH)
    logger.info("Source centroids saved successfully")    

    if GENERATE_VISUALIZATION:
        sources = [m.get("source", "unknown") for m in metadatas]
        visualize_clusters(np.array(embeddings), sources)    

def visualize_clusters(X, sources):
    """Generate a 2D visualization using t-SNE."""
    logger.info("Generating cluster visualization... (this might take a few seconds)")

    # t-SNE reduces 384-dimensional embeddings down to 2 dimensions for plotting
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    # Unique color IDs for plotting
    unique_sources = list(set(sources))
    source_to_id = {src: i for i, src in enumerate(unique_sources)}
    color_ids = [source_to_id[src] for src in sources]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_ids, cmap='tab10', alpha=0.6)

    # Add legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_sources, title="Sources")
    plt.title("Document Embeddings by Source")

    save_path = os.path.join(CLUSTERS_DIR, "cluster_visualization.png")
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Visualization saved as {save_path}")

def predict_source(query_embedding):
    """Predict the source for a query by finding the nearest document centroid."""
    if not os.path.exists(CENTROIDS_PATH):
        raise FileNotFoundError("Centroids model not found. Train it first.")

    centroids = joblib.load(CENTROIDS_PATH)
    query_vec = np.array(query_embedding)
    
    # Find the source with minimum Euclidean distance to the query vector
    best_source = None
    min_dist = float("inf")
    
    for source, centroid in centroids.items():
        dist = np.linalg.norm(query_vec - centroid)
        if dist < min_dist:
            min_dist = dist
            best_source = source

    return best_source if best_source else "none"
