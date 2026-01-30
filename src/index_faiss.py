import numpy as np
import faiss

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)
