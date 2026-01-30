import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

def compute_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_name)

    all_vecs: list[np.ndarray] = []
    rng = range(0, len(texts), batch_size)
    iterator = tqdm(rng, desc="Embedding batches") if show_progress else rng

    for i in iterator:
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False)
        all_vecs.append(np.asarray(vecs))

    embeddings = np.vstack(all_vecs).astype(np.float32)
    embeddings = normalize(embeddings, axis=1, norm="l2").astype(np.float32)
    return embeddings

def embed_query(query: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([query])
    vec = normalize(vec, axis=1, norm="l2").astype(np.float32)
    return vec

