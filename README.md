# Movie Semantic Search (Local)

A small local-only semantic search demo for movies.  
It encodes movie overviews into embeddings, builds a FAISS index, and returns the Top-K most similar movies for a user query.

This repo is meant for learning and for a reproducible GitHub project. No paid APIs, no deployment required.

---

## What you can do

- Build embeddings from a CSV dataset (one-time step)
- Create a FAISS index for fast similarity search
- Query from:
  - a CLI script
  - a local Streamlit web UI

---

## How it works (short version)

1. **Load & clean data**  
   Read `tmdb-movies.csv`, keep the minimal columns needed for search and display (title, overview, year, etc.).

2. **Embedding**  
   Use `sentence-transformers` (`all-MiniLM-L6-v2`) to convert each overview into a vector.  
   Vectors are **L2-normalized**.

3. **Indexing**  
   Build `faiss.IndexFlatIP`. With normalized vectors, **inner product ranking is equivalent to cosine similarity ranking**.

4. **Query**  
   Encode the query → FAISS `search()` → get Top-K indices → map indices back to rows in the metadata table.

---

## Repository layout

```text
data/
  tmdb-movies.csv                 # input data
src/                              # core modules (embedding, indexing, search)
scripts/
  build_assets.py                 # build embeddings + FAISS index (one-time)
  query_cli.py                    # CLI search
app_streamlit.py                  # local UI (optional)
requirements.txt


Generated files (not committed)

The following files are created by build_assets.py and should not be committed:

data/movie_embeddings.npy

data/faiss.index

data/meta.parquet
