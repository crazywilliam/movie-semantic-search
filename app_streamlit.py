import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH 
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import Paths, ModelConfig
from src.index_faiss import load_faiss_index
from src.search import search

@st.cache_resource
def load_assets():
    paths = Paths()
    cfg = ModelConfig()
    meta = pd.read_parquet(paths.meta_path)
    index = load_faiss_index(str(paths.faiss_index_path))
    return meta, index, cfg.embedding_model_name

st.set_page_config(page_title="Movie Semantic Search", layout="wide")
st.title("🎬 Movie Semantic Search (Local, Free)")

meta, index, model_name = load_assets()

query = st.text_input(
    "Describe the movie you want:",
    placeholder='e.g., "NASA tried to rescue an astronaut stranded on Mars"'
)
k = st.slider("Top-k results", min_value=1, max_value=20, value=5)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        df = search(query=query, k=k, index=index, meta=meta, model_name=model_name)

    st.success(f"Found {len(df)} results")
    for _, row in df.iterrows():
        st.markdown(f"### {int(row['rank'])}. {row['title']} ({row['release_year']})")
        st.write(f"Similarity: {row['similarity']:.4f}")
        st.write(row["overview"])
        st.divider()
