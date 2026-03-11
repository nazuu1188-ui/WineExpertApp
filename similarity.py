import numpy as np
import pandas as pd

def normalize_id(x) -> str:
    if x is None:
        return ""
    return str(x).replace(" ", "").strip()

def cosine_similarity_top_k(df: pd.DataFrame, id_col: str, target_id: str, feature_cols: list[str], k: int = 10) -> pd.DataFrame:
    if df.empty or not feature_cols:
        return pd.DataFrame()

    df2 = df.copy()
    df2[id_col] = df2[id_col].map(normalize_id)
    target_id = normalize_id(target_id)

    idx = df2.index[df2[id_col] == target_id]
    if len(idx) == 0:
        return pd.DataFrame()

    i = int(idx[0])
    X = df2[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X = np.nan_to_num(X, nan=0.0)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-9, None)

    sims = (Xn @ Xn[i]).reshape(-1)
    order = np.argsort(-sims)
    order = order[order != i][:k]

    out = df2.loc[order].copy()
    out.insert(1, "Similarity", sims[order])
    return out