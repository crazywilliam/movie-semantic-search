import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd

from src.config import Paths, ModelConfig
from src.index_faiss import load_faiss_index
from src.search import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie semantic search (local + free)")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Search query text")
    parser.add_argument("--k", dest="k", type=int, default=5, help="Top-k results")
    args = parser.parse_args()

    paths = Paths()
    cfg = ModelConfig()

    meta = pd.read_parquet(paths.meta_path)
    index = load_faiss_index(str(paths.faiss_index_path))

    results = search(
        query=args.query,
        k=args.k,
        index=index,
        meta=meta,
        model_name=cfg.embedding_model_name,
    )

    print(results[["rank", "title", "similarity", "release_year"]].to_string(index=False))

if __name__ == "__main__":
    main()
