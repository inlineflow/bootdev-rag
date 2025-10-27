from collections import defaultdict, Counter
import math
import os
from typing import Counter as CounterType, Dict, List, Set
import pickle

from search_utils import BM25_K1
from tokens import Movie, load_movies, preprocess, remove_stopwords, strip_punctuation, tokenize


class InvertedIndex:
    index: Dict[str, Set[int]]
    docmap: Dict[int, Movie]
    term_frequencies: Dict[int, CounterType]

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.docmap = {}

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = preprocess(text)
        for t in tokens:
            self.index[t].add(doc_id)
            self.term_frequencies[doc_id][t] += 1
            # self.index.setdefault(t, set()).add(doc_id)
            # self.term_frequencies.setdefault(doc_id, Counter())[t]+=1

            # if t not in self.index:
            #     self.index[t] = set([doc_id])
            #     continue
            #
            # self.index[t] = set([*self.index[t], doc_id])

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
        # x = Path("cache/")
        cache_path = os.path.join(os.getcwd(), "cache")
        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")
        term_frequencies_path = os.path.join(cache_path, "term_frequencies.pkl")
        os.makedirs(cache_path, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) ->None:
        cache_path = os.path.join(os.getcwd(), "cache")
        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")
        term_frequencies_path = os.path.join(cache_path, "term_frequencies.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} doesn't exist")

        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"{docmap_path} doesn't exist")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

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

    def get_bm25_tf(self, doc_id:int, term:str, k1: float = BM25_K1) -> float:
        base_tf = self.get_tf(doc_id, term)
        bm25_tf = (base_tf * (k1 + 1)) / (base_tf + k1)
        
        return bm25_tf

