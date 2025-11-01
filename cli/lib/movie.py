from dataclasses import dataclass
import json
from typing import List


@dataclass
class Movie:
    id: int
    title: str
    description: str


def load_movies() -> List[Movie]:
    with open("data/movies.json") as f:
        data = json.load(f)
        movies = [Movie(**movie_dict) for movie_dict in data.get("movies", [])]
        return movies
