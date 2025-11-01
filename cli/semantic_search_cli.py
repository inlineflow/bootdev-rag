#!/usr/bin/env python3

import argparse
from lib.movie import load_movies
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    verify_embeddings,
    verify_model,
    embed_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_cmd = subparsers.add_parser("verify")
    embed_text_cmd = subparsers.add_parser("embed_text")
    embed_text_cmd.add_argument("text", type=str)

    verify_embeddings_cmd = subparsers.add_parser("verify_embeddings")

    embedquery_cmd = subparsers.add_parser("embedquery")
    embedquery_cmd.add_argument("query", type=str)

    search_cmd = subparsers.add_parser("search")
    search_cmd.add_argument("query", type=str)
    search_cmd.add_argument("--limit", default=5, type=int, help="Optional amount of records to return. Returns all the records otherwise.")


    chunk_cmd = subparsers.add_parser("chunk")
    chunk_cmd.add_argument("text", type=str)
    chunk_cmd.add_argument("--chunk-size", default=200, type=int, help="Number of words in a chunk")
    chunk_cmd.add_argument("--overlap", default=0, type=int, help="Number of words shared between neighboring chunks")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            s = SemanticSearch()
            docs = load_movies()
            s.load_or_create_embeddings(docs)
            res = s.search(args.query, args.limit)
            for i, k in enumerate(res):
                score = k[0]
                doc = k[1]
                print(f"{i+1}. {doc.title} (score: {score:.4f})")
                print(doc.description)
                print()
            # print(res)
        case "chunk":
            text = args.text.split()
            size = args.chunk_size
            overlap = args.overlap
            i = 0
            count = 0
            chunks = []
            while count < len(text):
                chunk = text[i * size:(i + 1) * size - overlap]
                print(chunk)
                chunks.append(chunk)
                i += 1
                count += len(chunk)
            

            print(f"Chunking {len(args.text.strip())} characters") 
            for index, c in enumerate(chunks):
                print(f"{index + 1}. {" ".join(c)}")

                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
