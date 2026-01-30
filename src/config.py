from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"

    csv_path: Path = data_dir / "tmdb-movies.csv"
    embeddings_path: Path = data_dir / "movie_embeddings.npy"
    faiss_index_path: Path = data_dir / "faiss.index"
    meta_path: Path = data_dir / "meta.parquet"

@dataclass(frozen=True)
class ModelConfig:
    embedding_model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
