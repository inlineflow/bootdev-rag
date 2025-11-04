#!/usr/bin/env python3

import argparse
from json import load
import re
from lib.movie import load_movies
from lib.semantic_search import (
    ChunkedSemanticSearch,
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

    semantic_chunk_cmd = subparsers.add_parser("semantic_chunk")
    semantic_chunk_cmd.add_argument("text", type=str)
    semantic_chunk_cmd.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    semantic_chunk_cmd.add_argument("--overlap", type=int, default=0)

    embed_chunks_cmd = subparsers.add_parser("embed_chunks")

    search_chunked_cmd = subparsers.add_parser("search_chunked")
    search_chunked_cmd.add_argument("text", type=str)
    search_chunked_cmd.add_argument("--limit", type=int, default=5)


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
            words = args.text.split()
            size = args.chunk_size
            overlap = args.overlap
            i = 0
            n_sentences = len(words)
            chunks = []
            while i < n_sentences - overlap:
                chunk = words[i:i+size]
                chunks.append(chunk)
                i += size - overlap
            

            print(f"Chunking {len(args.text.strip())} characters") 
            for index, c in enumerate(chunks):
                print(f"{index + 1}. {" ".join(c)}")

        case "semantic_chunk":
            sentences = re.split(r"(?<=[.!?])\s+", args.text)
            size = args.max_chunk_size
            overlap = args.overlap
            i = 0
            n_sentences = len(sentences)
            chunks = []
            while i < n_sentences - overlap:
                chunk = sentences[i:i+size]
                chunks.append(chunk)
                i += size - overlap

            print(f"Semantically chunking {len(args.text.strip())} characters") 
            for index, c in enumerate(chunks):
                print(f"{index + 1}. {" ".join(c)}")
                
        case "embed_chunks":
            movies = load_movies()
            cs = ChunkedSemanticSearch()
            embeddings = cs.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = load_movies()
            cs = ChunkedSemanticSearch()
            embeddings = cs.load_or_create_chunk_embeddings(movies)
            # print(len(embeddings))
            # print(cs.chunk_metadata)
            results = cs.search_chunks(args.text, args.limit)
            for i, r in enumerate(results):
                TITLE = r.get("title")
                SCORE = r.get("score")
                DESCRIPTION = r.get("document")
                print(f"\n{i+1}. {TITLE} (score: {SCORE:.4f})")
                print(f"   {DESCRIPTION}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
