#!/usr/bin/env python3

import argparse
from typing import List
from search_utils import BM25_B, BM25_K1
from lib.keyword_search import InvertedIndex
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

    idf_parse = subparsers.add_parser(
        "idf", help="Get the inverse document frequency for the speciifed term"
    )
    idf_parse.add_argument("term", type=str, help="Term")

    tfidf_parse = subparsers.add_parser(
        "tfidf", help="Get term frequency for specified term in the specified document"
    )
    tfidf_parse.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parse.add_argument("term", type=str, help="Term")

    bm25_idf_command = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_command.add_argument("term", type=str, help="Term")

    bm25_tf = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given term")
    bm25_tf.add_argument("doc_id", type=int, help="doc id")
    bm25_tf.add_argument("term", type=str, help="Term")
    bm25_tf.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Saturation constant, used for tuning")
    bm25_tf.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25_search = subparsers.add_parser("bm25search", help="Performs BM25 search for a given doc_id and term")
    bm25_search.add_argument("query", type=str, help="Search query")
    
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
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "tf":
            index = InvertedIndex()
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverted document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            idf = index.get_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            index = InvertedIndex()
            index.load()
            items = index.bm25_search(args.query, 5)
            for m, score in items:
                print(f"({m.id}) {m.title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
