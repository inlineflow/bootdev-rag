from dataclasses import dataclass
import json
from typing import List
import string
from nltk.stem import PorterStemmer

@dataclass
class Movie:
    id: int
    title: str
    description: str


def load_movies() -> List[Movie]:
    with open("data/movies.json") as f:
        data = json.load(f)
        movies = [
                Movie(**movie_dict) for movie_dict in data.get("movies", [])
                ]
        return movies

def tokenize(s: str) -> List[str]:
    result = list(filter(lambda i: i != "", s.split()))
    return result

def strip_punctuation(s: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)

def load_stopwords() -> List[str]:
    with open("data/stopwords.txt") as f:
        result = f.read().splitlines()
        return result

def remove_stopwords(tokens: List[str]) -> List[str]:
    stopwords = load_stopwords()
    result = set(tokens).difference(stopwords)
    return list(result)

# def stem(tokens: List[str]) -> List[str]:
#     stemmer = PorterStemmer()
#     return list(map(stemmer.stem, tokens))

def preprocess(d: str) -> List[str]:
    # transformers = [str.lower, strip_punctuation, tokenize, remove_stopwords, stem]
    
    stemmer = PorterStemmer()
    val = d.lower()
    
    val = strip_punctuation(val)

    tokens = tokenize(val)
    no_stopwords = remove_stopwords(tokens)
    stemmed = list(map(stemmer.stem, no_stopwords))

    return stemmed
