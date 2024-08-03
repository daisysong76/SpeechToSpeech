import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # Load the saved embeddings
# embedding = torch.load('/data/daisysxm76/speechtospeech/dataset_fr_en/embeddings/common_voice_fr_17299508.mp3.pt')

# # Inspect the shape and content of the embedding
# print(embedding.shape)
# print(embedding)

def load_and_verify_embeddings(embeddings_dir):
    embedding_files = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith(".pt")]
    
    for embedding_file in embedding_files:
        embeddings = torch.load(embedding_file)
        
        # Check for NaN values
        has_nan = torch.isnan(embeddings).any()
        print(f"{embedding_file} contains NaN: {has_nan}")

        # Check for infinite values
        has_inf = torch.isinf(embeddings).any()
        print(f"{embedding_file} contains Inf: {has_inf}")

        # Check statistical properties
        mean = embeddings.mean().item()
        std = embeddings.std().item()
        min_val = embeddings.min().item()
        max_val = embeddings.max().item()

        print(f"{embedding_file} - Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")

        # Flatten the embeddings for visualization if necessary
        flattened_embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()

        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(flattened_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title(f'PCA of {embedding_file}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        # Reduce dimensions using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(flattened_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
        plt.title(f't-SNE of {embedding_file}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

if __name__ == "__main__":
    embeddings_dir = "/data/daisysxm76/speechtospeech/dataset_fr_en/embeddings/"
    load_and_verify_embeddings(embeddings_dir)
