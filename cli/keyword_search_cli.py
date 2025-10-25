#!/usr/bin/env python3

import argparse
from typing import List
from inv_index import InvertedIndex
from tokens import Movie, preprocess


# def keyword_search(query: str, index: InvertedIndex):
#     q = preprocess(query)
#     movies = load_movies()
#     result:List[Movie] = []
#     for m in movies:
#         q_set = set(q)
#         title_tokens_set = set(preprocess(m.title))
#         for qt in q_set:
#             success = list(filter(lambda tt: qt in tt, title_tokens_set))
#             if len(success) != 0:
#                 # print("success: ", success)
#                 # print("query: ", q_set)
#                 result.append(m)
#                 break
#
#     f_result = sorted(result, key=lambda item: item.id)[:5]
#     for i, r in enumerate(f_result):
#         print(f"{i+1}. {r.title}")
#


def keyword_search(query: str, index: InvertedIndex):
    q = preprocess(query)
    result: List[Movie] = []
    for qtoken in q:
        ids = index.get_documents(qtoken)
        for id in ids:
            result.append(index.docmap[id])

        if len(result) >= 5:
            break


    f_result = sorted(result, key=lambda item: item.id)[:5]
    for i, r in enumerate(f_result):
        print(f"{i+1}. {r.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for specified term in the specified document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                index = InvertedIndex()
                index.load()
                keyword_search(args.query, index)
            except Exception as e:
                print(e)
                exit()

            pass
        case "build":
            pass
            index = InvertedIndex()
            index.build()
            index.save()
        case "tf":
            index = InvertedIndex()
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
