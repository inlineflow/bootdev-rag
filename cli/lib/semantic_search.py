import os
import re
import json
from typing import Any, Dict, List, Tuple, TypedDict
from sentence_transformers import SentenceTransformer
import numpy as np

from lib.movie import Movie, load_movies

SCORE_PRECISION = 4

class SemanticSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: Dict[str, Any]

class SemanticSearch:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map:Dict[int, Movie] = {}
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

            if len(self.embeddings) == len(self.documents):
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

def chunk_semantically(text:str, chunk_size: int, overlap: int) -> List[str]:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if sentences == []:
                return []

            sentences = [s.strip() for s in sentences]
            i = 0
            chunks = []
            while i < len(sentences):
                chunk_sentences = sentences[i:i+chunk_size]
                if not chunk_sentences:
                    break
                chunk = " ".join(chunk_sentences).strip()
                if chunk:
                    chunks.append(chunk)
                i += max(1, chunk_size - overlap)

            return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = 'all-MiniLM-L6-v2') -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.embeddings_cache_path = os.path.join(
            self.cache_path, "chunk_embeddings.npy"
        )
        self.metadata_cache_path = os.path.join(
            self.cache_path, "chunk_metadata.json"
        )

    def build_chunk_embeddings(self, documents:List[Movie]):
        self.documents = documents
        all_chunks:List[str] = []
        metadata:List[Dict] = []
        for movie_idx, doc in enumerate(documents):
            self.document_map[doc.id] = doc
            if doc.description == "":
                continue

            chunks = chunk_semantically(doc.description, 4, 1)
            all_chunks.extend(chunks)
            for chunk_idx, _ in enumerate(chunks):
                meta = {"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)}
                metadata.append(meta)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata


        with open(self.embeddings_cache_path, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(self.metadata_cache_path, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: List[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc

        
        if os.path.exists(self.embeddings_cache_path) and os.path.exists(self.metadata_cache_path):
            print("Loading cache...")
            with open(self.embeddings_cache_path, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(self.metadata_cache_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]


            print(self.chunk_embeddings.shape[0], len(self.chunk_metadata))
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query:str, limit:int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Embeddings are not loaded. Load or build embeddings first")
        q = self.generate_embedding(query)
        chunk_scores = []

        for i in range(len(self.chunk_embeddings)):
            e = self.chunk_embeddings[i]
            cmeta = self.chunk_metadata[i]
            score = cosine_similarity(q, e)
            chunk_scores.append(
                     {
                        "movie_idx": cmeta["movie_idx"],
                        "chunk_idx": cmeta["chunk_idx"],
                        "score": score
                     })

        score_map: Dict[int, float] = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"] 
            new_score = cs["score"]
            if movie_idx not in score_map or new_score > score_map[movie_idx]:
                score_map[movie_idx] = new_score

        scores_sorted = sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:limit]
        results: List[SemanticSearchResult] = []
        for ss in scores_sorted:
            movie_idx = ss[0]
            doc = self.documents[movie_idx]
            doc_id = doc.id
            score = ss[1]
            movie = self.document_map[doc_id]
            title = movie.title
            document = movie.description
            metadata = {}
            item: SemanticSearchResult = {
              "id": doc_id,
              "title": title,
              "document": document[:100],
              "score": round(score, SCORE_PRECISION),
              "metadata": metadata or {}
            }
            results.append(item)

        return results

