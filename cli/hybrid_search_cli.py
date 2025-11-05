import argparse

from lib.movie import load_movies
from lib.hybrid_search import HybridSearch
from lib.hybrid_search import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_cmd = subparsers.add_parser("normalize")
    normalize_cmd.add_argument("values", type=float, nargs="+")

    weighted_search_cmd = subparsers.add_parser("weighted-search")
    weighted_search_cmd.add_argument("query", type=str)
    weighted_search_cmd.add_argument("--alpha", type=float, default=0.5)
    weighted_search_cmd.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    print(args)

    match args.command:
        case "normalize":
            values = normalize(args.values)
            for score in values:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            hs.weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
