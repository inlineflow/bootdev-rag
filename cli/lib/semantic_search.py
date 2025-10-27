from sentence_transformers import SentenceTransformer


class SemanticSearch():
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text:str):
        if not text.strip():
            raise ValueError("Query must not be empty")

        embedding = self.model.encode([text])
        return embedding[0]

def verify_model():
    s = SemanticSearch()
    print(f"Model loaded: {s.model}")
    print(f"Max sequence length: {s.model.max_seq_length}")

def embed_text(text:str):
    s = SemanticSearch()
    embedding = s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
