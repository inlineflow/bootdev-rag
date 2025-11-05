import os
from typing import List

from .movie import Movie
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
import itertools

def normalize(values: list[float]) -> list[float]:
    if len(values) == 0:
        return []
    high = max(values)
    low = min(values)
    if high == low:
        return list(itertools.repeat(1.0, len(values)))

    k = high - low
    result = [(score - low) / k for score in values]
    return result

class HybridSearch():
    def __init__(self, documents: List[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm_25_search(self, query:str, limit:int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _semantic_search(self, query:str, limit:int):
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query:str, alpha:float, limit:int=5):
        print("Running weighted search...")
        search_limit = limit * 500
        bm25 = self._bm_25_search(query, search_limit)
        semantic = self._semantic_search(query, search_limit)
        bm25_scores = [score for _, score in bm25]
        maxbm25 = max(bm25_scores)
        minbm25 = min(bm25_scores)
        semantic_scores = [i["score"] for i in semantic]
        max_semantic_score = max(semantic_scores)
        # for i in range(len(bm25)):
        for i in range(5):
            keyword_item = bm25[i]
            item_id = keyword_item[0].id 
            semantic_item = next((i for i in semantic if i["id"] == keyword_item[0].id), None)
            if semantic_item is None:
                print(f"Item with id: {item_id} not found in semantic search results.")
                continue

            

            
            # print(keyword_item[0].id)
            # print(semantic_item["id"])

    def rrf_search(self, query:str, k:float, limit:int=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

