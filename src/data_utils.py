import pandas as pd

REQUIRED_COLS = ["title", "overview", "release_date", "poster_path"]

def load_movies_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    keep_cols = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep_cols].copy()

    df["overview"] = df["overview"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    df = df[df["overview"].str.len() > 0].reset_index(drop=True)

    if "release_date" in df.columns:
        df["release_year"] = df["release_date"].astype(str).str.slice(0, 4)
    else:
        df["release_year"] = ""

    return df

def build_meta_table(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["title", "overview", "release_year"]
    if "poster_path" in df.columns:
        meta_cols.append("poster_path")
    return df[meta_cols].copy()
