from collections import defaultdict, Counter
import math
import os
from typing import Counter as CounterType, Dict, List, Set, Tuple
import pickle

from search_utils import BM25_B, BM25_K1
from tokens import Movie, load_movies, preprocess
from functools import reduce


class InvertedIndex:
    index: Dict[str, Set[int]]
    docmap: Dict[int, Movie]
    term_frequencies: Dict[int, CounterType]
    doc_lengths: Dict[int, int]

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.docmap = {}
        self.doc_lengths = {}
        self.cache_path = os.path.join(os.getcwd(), "cache")
        self.index_path = os.path.join(self.cache_path, "index.pkl")
        self.docmap_path = os.path.join(self.cache_path, "docmap.pkl")
        self.term_frequencies_path = os.path.join(self.cache_path, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(self.cache_path, "doc_lengths.pkl")

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = preprocess(text)
        for t in tokens:
            self.index[t].add(doc_id)
            self.term_frequencies[doc_id][t] += 1

        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term:str) -> List[int]:
        t = preprocess(term)[0]
        if t not in self.index:
            return []

        result = sorted(list(self.index[t]))
        return result

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            content = f"{m.title} {m.description}"
            self.__add_document(m.id, content)
            self.docmap[m.id] = m

    def save(self) -> None:
        os.makedirs(self.cache_path, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
            # self.doc_lengths = pickle.load(f)

    def load(self) ->None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"{self.index_path} doesn't exist")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"{self.docmap_path} doesn't exist")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id:int, term:str) -> int:
        token = preprocess(term)
        if len(token) > 1:
            raise ValueError(f"Provided term must have exactly 1 token, actual term: {token}")

        return self.term_frequencies[doc_id][token[0]]

    def get_idf(self, term:str) -> float:
        query = preprocess(term)
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[query[0]])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        t = preprocess(term)
        if len(t) != 1:
            raise ValueError("Term must be a single token")

        q = t[0]
        df = len(self.index.get(q, set()))
        N = len(self.docmap)
        if term == "grizzly":
            print(df)
            print(self.index[q])
            print(N)
            print(t)
        bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return bm25

    def __get_average_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id:int, term:str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        base_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_average_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_tf = (base_tf * (k1 + 1)) / (base_tf + k1 * length_norm)
        
        return bm25_tf

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query: str, limit: int) -> List[Tuple[Movie, float]]:
        q = preprocess(query)
        scores:Dict[int, float] = {}

        for doc_id, m in self.docmap.items():
            score = reduce(lambda accumulator, term: accumulator + self.bm25(doc_id, term), q, 0.0)
            scores[doc_id] = score

        result = sorted(scores.items(), reverse=True, key=lambda i: i[1])[:limit]
        # print(result)
        r = [(self.docmap[d[0]], d[1]) for d in result]
        # print(r)
        return r

