import os
from typing import Dict, List, Set
import pickle

from tokens import Movie, load_movies, preprocess


class InvertedIndex:
    index: Dict[str, Set[int]]
    docmap: Dict[int, Movie]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = preprocess(text)
        for t in tokens:
            self.index.setdefault(t, set()).add(doc_id)
            # if t not in self.index:
            #     self.index[t] = set([doc_id])
            #     continue
            #
            # self.index[t] = set([*self.index[t], doc_id])

    def get_documents(self, term:str) -> List[int]:
        t = term.lower()
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
        os.makedirs(cache_path, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) ->None:
        cache_path = os.path.join(os.getcwd(), "cache")
        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} doesn't exist")

        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"{docmap_path} doesn't exist")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
