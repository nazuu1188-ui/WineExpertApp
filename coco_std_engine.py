import numpy as np
import pandas as pd

def _pav(y, w):
    y = y.astype(float)
    w = w.astype(float)
    blocks = []
    for i in range(len(y)):
        blocks.append([i, i, y[i], w[i]])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            new_w = b1[3] + b2[3]
            new_mean = (b1[2] * b1[3] + b2[2] * b2[3]) / max(new_w, 1e-12)
            blocks.append([b1[0], b2[1], new_mean, new_w])

    y_hat = np.empty_like(y, dtype=float)
    for start, end, mean, _ in blocks:
        y_hat[start:end+1] = mean
    return y_hat

def _isotonic_fit_by_rank(rank: np.ndarray, target: np.ndarray, decreasing: bool = True) -> dict:
    df = pd.DataFrame({"r": rank, "t": target}).dropna()
    g = df.groupby("r", dropna=False).agg(mean=("t", "mean"), w=("t", "size")).reset_index()
    g = g.sort_values("r")

    y = g["mean"].to_numpy(dtype=float)
    w = g["w"].to_numpy(dtype=float)

    if decreasing:
        y_fit = -_pav(-y, w)  # nonincreasing
    else:
        y_fit = _pav(y, w)    # nondecreasing

    return dict(zip(g["r"].tolist(), y_fit.tolist()))

def coco_std_backfit(rank_df: pd.DataFrame, x_a_cols: list[str], y_col: str = "A12", n_iter: int = 25) -> pd.DataFrame:
    y = pd.to_numeric(rank_df[y_col], errors="coerce").to_numpy(dtype=float)
    n = len(y)
    intercept = np.nanmean(y)

    comps = {a: np.zeros(n, dtype=float) for a in x_a_cols}

    for _ in range(n_iter):
        total = intercept + sum(comps[a] for a in x_a_cols)
        for a in x_a_cols:
            resid = y - (total - comps[a])
            r = pd.to_numeric(rank_df[a], errors="coerce").to_numpy(dtype=float)
            mapping = _isotonic_fit_by_rank(r, resid, decreasing=True)

            fitted = np.array([mapping.get(rv, np.nan) for rv in r], dtype=float)
            fitted = np.nan_to_num(fitted, nan=0.0)

            m = fitted.mean()
            comps[a] = fitted - m
            intercept += m

    y_hat = intercept + sum(comps[a] for a in x_a_cols)

    out = pd.DataFrame({"OAM": rank_df["OAM"].astype(str)})
    for a in x_a_cols:
        out[f"X({a})"] = comps[a]

    out["Becslés"] = y_hat
    out["Tény+0"] = y
    out["Delta"] = out["Tény+0"] - out["Becslés"]
    out["Delta/Tény"] = np.where(out["Tény+0"] != 0, 100.0 * out["Delta"] / out["Tény+0"], np.nan)
    return out

def quantize_components(df: pd.DataFrame, step: float = 0.4, decimals: int = 1) -> pd.DataFrame:
    out = df.copy()
    xcols = [c for c in out.columns if c.startswith("X(A")]
    for c in xcols:
        v = out[c].astype(float).to_numpy()
        vq = np.round(v / step) * step
        out[c] = np.round(vq, decimals)

    out["Becslés"] = np.round(out["Becslés"].astype(float), decimals)
    out["Delta"] = np.round(out["Delta"].astype(float), decimals)
    out["Delta/Tény"] = np.round(out["Delta/Tény"].astype(float), 0)
    return out

def inverse_validation(normal_df: pd.DataFrame, rank_df: pd.DataFrame, x_a_cols: list[str]) -> pd.Series:
    inv_rank = rank_df.copy()
    for a in x_a_cols:
        m = pd.to_numeric(inv_rank[a], errors="coerce").max()
        inv_rank[a] = (m + 1) - pd.to_numeric(inv_rank[a], errors="coerce")

    inv_df = coco_std_backfit(inv_rank, x_a_cols=x_a_cols, y_col="A12", n_iter=25)

    nd = pd.to_numeric(normal_df["Delta"], errors="coerce")
    id_ = pd.to_numeric(inv_df["Delta"], errors="coerce")
    return ((nd * id_) <= 0).astype(int)