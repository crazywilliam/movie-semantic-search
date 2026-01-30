import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

from src.config import Paths, ModelConfig
from src.data_utils import load_movies_csv, build_meta_table
from src.embed import compute_embeddings
from src.index_faiss import build_faiss_index, save_faiss_index

def main() -> None:
    paths = Paths()
    cfg = ModelConfig()

    print(f"Loading data from: {paths.csv_path}")
    df = load_movies_csv(str(paths.csv_path))
    meta = build_meta_table(df)

    print(f"Movies: {len(meta)}")
    print("Computing embeddings (one-time)...")
    embeddings = compute_embeddings(
        texts=meta["overview"].tolist(),
        model_name=cfg.embedding_model_name,
        batch_size=cfg.batch_size,
        show_progress=True,
    )

    print(f"Saving embeddings to: {paths.embeddings_path}")
    np.save(paths.embeddings_path, embeddings)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving FAISS index to: {paths.faiss_index_path}")
    save_faiss_index(index, str(paths.faiss_index_path))

    print(f"Saving metadata to: {paths.meta_path}")
    meta.to_parquet(paths.meta_path, index=False)

    print("✅ Done. You can now run CLI search.")

if __name__ == "__main__":
    main()

