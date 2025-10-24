#!/usr/bin/env python3

import argparse, json, string
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Movie:
    id: int
    title: str
    description: str

@dataclass
class Data:
    movies: List[Movie]


def load_movies() -> List[Movie]:
    with open("data/movies.json") as f:
        data = json.load(f)
        movies = [
                Movie(**movie_dict) for movie_dict in data.get("movies", [])
                ]
        return movies

def tokenize(s: str) -> List[str]:
    result = list(filter(lambda i: i != "", s.split(" ")))
    return result

def strip_punctuation(s: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)

def prep_query(q: str) -> List[str]:
    return tokenize(q.lower())

def preprocess(d: str) -> List[str]:
    transformers = [str.lower, strip_punctuation]
    val = d
    for t in transformers:
        val = t(val)

    result = tokenize(val)

    return result

def keyword_search(query: str):
    q = prep_query(query)
    movies = load_movies()
    result:List[Movie] = []
    for m in movies:
        q_set = set(q)
        title_tokens_set = set(preprocess(m.title))
        for qt in q_set:
            success = list(filter(lambda tt: qt in tt, title_tokens_set))
            if len(success) != 0:
                print("success: ", success)
                print("query: ", q_set)
                result.append(m)
                break

    f_result = sorted(result, key=lambda item: item.id)
    for i, r in enumerate(f_result):
        print(f"{i+1}. {r.title}")

    # print(x[0:5])



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            keyword_search(args.query)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
