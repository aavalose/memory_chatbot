from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def embed_mini(documents: str):
    """Generate embeddings for documents using MiniLM model.
    
    Args:
        documents (str): Documents to embed
        
    Returns:
        numpy.ndarray: Document embeddings
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(documents)


def get_top_k_similarities(new_query_embedding, corpus_embeddings, k=3):
    """Get indices of top-k most similar documents."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    
    # Ensure new_query_embedding is a 2D array
    if new_query_embedding.ndim == 1:
        new_query_embedding = new_query_embedding.reshape(1, -1)
    
    # Calculate similarities
    similarities = np.squeeze(cosine_similarity(new_query_embedding, corpus_embeddings)).flatten()

    num_documents = len(similarities)
    if num_documents == 0:
        raise ValueError("Corpus embeddings are empty, cannot compute similarities.")

    k = min(k, num_documents)  # Ensure k does not exceed available documents
    if k == 0:
        return np.array([], dtype=int)

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Convert tensor to NumPy if it's a PyTorch tensor
    if isinstance(top_k_indices, torch.Tensor):
        top_k_indices = top_k_indices.cpu().numpy()  # Move to CPU and convert to numpy

    # Debugging print statements
    print("Top-k indices:", top_k_indices)
    print("Type of top_k_indices:", type(top_k_indices))
    print("Size of top_k_indices:", top_k_indices.size if isinstance(top_k_indices, np.ndarray) else "N/A")

    # Sort them by similarity score if they exist
    if isinstance(top_k_indices, np.ndarray) and top_k_indices.size > 0:
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

    return top_k_indices.astype(int)


def get_top_k_similar_documents(new_query_embedding, corpus_embeddings, questions, qa_dict, k=5):
    """Get top-k most similar documents with their questions and answers.
    
    Args:
        new_query_embedding (numpy.ndarray): Query embedding
        corpus_embeddings (numpy.ndarray): Document embeddings
        questions (list): List of questions corresponding to embeddings
        qa_dict (dict): Dictionary mapping questions to answers
        k (int, optional): Number of results to return. Defaults to 5.
        
    Returns:
        list: List of tuples containing (question, answer, similarity_score) for top-k matches
    """
    top_k_indices = get_top_k_similarities(new_query_embedding, corpus_embeddings, k)
    scores = np.squeeze(new_query_embedding @ corpus_embeddings.T)
    
    results = []
    for idx in top_k_indices:
        idx_scalar = idx.item()  # Convert numpy integer to Python scalar
        question = questions[idx_scalar]
        answer = qa_dict[question]
        score = scores[idx_scalar]
        results.append((question, answer, score))
        
    return results
