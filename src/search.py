import pandas as pd
import faiss
from .embed import embed_query

def search(
    query: str,
    k: int,
    index: faiss.Index,
    meta: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    qvec = embed_query(query, model_name)
    scores, ids = index.search(qvec, k)

    rows = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        item = meta.iloc[int(idx)]
        rows.append(
            {
                "rank": rank,
                "title": item["title"],
                "similarity": float(score),
                "release_year": item.get("release_year", ""),
                "overview": item["overview"],
            }
        )

    return pd.DataFrame(rows)

