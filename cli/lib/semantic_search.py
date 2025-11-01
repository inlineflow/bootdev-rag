import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from lib.movie import Movie, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.cache_path = os.path.join(os.getcwd(), "cache")
        self.embeddings_cache_path = os.path.join(
            self.cache_path, "movie_embeddings.npy"
        )

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Query must not be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: List[Movie]):
        self.documents = documents
        data = []
        for doc in documents:
            self.document_map[doc.id] = doc
            data.append(f"{doc.title}: {doc.description}")

        x = self.model.encode(data, show_progress_bar=True)
        self.embeddings = x
        with open(self.embeddings_cache_path, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings
        # print(documents)

    def load_or_create_embeddings(self, documents: List[Movie]): 
        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc
        
        if os.path.exists(self.embeddings_cache_path):
            print("Loading cache...")
            with open(self.embeddings_cache_path, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(self.embeddings):
                print("Cache matched")
                return self.embeddings

        print("Cache not matched. Rebuilding embeddings...")
        return self.build_embeddings(documents)

    def search(self, query, limit) -> List[Tuple[float, Movie]]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        q_embed = self.generate_embedding(query)
        scores: List[float] = []
        for e in self.embeddings:
            score = cosine_similarity(q_embed, e)
            scores.append(score)
        
        x: List[Tuple[float, Movie]] = list(zip(scores, self.documents)) # type: ignore
        # print(x)
        result = sorted(x, key=lambda item: item[0], reverse=True)
        return result[:limit]


def verify_model():
    s = SemanticSearch()
    print(f"Model loaded: {s.model}")
    print(f"Max sequence length: {s.model.max_seq_length}")


def embed_text(text: str):
    s = SemanticSearch()
    embedding = s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    s = SemanticSearch()
    movies = load_movies()
    embeddings = s.load_or_create_embeddings(movies)
    
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    s = SemanticSearch()
    embedding = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

    
