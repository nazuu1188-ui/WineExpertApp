import re
import pandas as pd

WINE_X = [
    ("fixed acidity", "g/dm³"),
    ("volatile acidity", "g/dm³"),
    ("citric acid", "g/dm³"),
    ("residual sugar", "g/dm³"),
    ("chlorides", "g/dm³"),
    ("free sulfur dioxide", "mg/dm³"),
    ("total sulfur dioxide", "mg/dm³"),
    ("density", "g/dm³"),
    ("pH", "pH scale"),
    ("sulphates", "g/dm³"),
    ("alcohol", "% vol."),
]
WINE_Y = ("quality", "Score")

CANON_X = [n for n, _u in WINE_X]
CANON_Y = WINE_Y[0]
UNITS = {n: u for n, u in WINE_X} | {WINE_Y[0]: WINE_Y[1]}

ALIASES = {
    **{n: n for n in CANON_X + [CANON_Y]},
    **{n.replace(" ", "_"): n for n in CANON_X + [CANON_Y]},
    "fixedacidity": "fixed acidity",
    "volatileacidity": "volatile acidity",
    "residualsugar": "residual sugar",
    "freesulfurdioxide": "free sulfur dioxide",
    "totalsulfurdioxide": "total sulfur dioxide",
}

def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def load_raw_table(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_excel(uploaded_file, sheet_name="Raw Data")
    except Exception:
        return pd.read_excel(uploaded_file, sheet_name=0)

def coerce_wine_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    rename = {}
    for c in cols:
        n1 = _norm(c)
        n2 = n1.replace(" ", "_")
        n3 = n1.replace(" ", "")
        if n1 in ALIASES:
            rename[c] = ALIASES[n1]
        elif n2 in ALIASES:
            rename[c] = ALIASES[n2]
        elif n3 in ALIASES:
            rename[c] = ALIASES[n3]

    df2 = df.rename(columns=rename)
    missing = [c for c in (CANON_X + [CANON_Y]) if c not in df2.columns]
    if missing:
        raise ValueError(
            "Missing required wine columns:\n- " + "\n- ".join(missing) +
            "\n\nFix: rename RAW headers to canonical names (e.g., 'fixed acidity', ..., 'quality')."
        )
    return df2

def build_attribute_meta(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[CANON_Y], errors="coerce")
    rows = []
    for i, col in enumerate(CANON_X, start=1):
        x = pd.to_numeric(df[col], errors="coerce")
        corr = x.corr(y)
        corr = float(corr) if pd.notna(corr) else 0.0
        direction_id = 0 if corr >= 0 else 1
        rows.append({
            "AttributeID": f"A{i}",
            "Attribute": col,
            "Unit": UNITS.get(col, ""),
            "CorrelationWithY": corr,
            "DirectionID": direction_id,
        })
    rows.append({
        "AttributeID": "A12",
        "Attribute": CANON_Y,
        "Unit": UNITS.get(CANON_Y, ""),
        "CorrelationWithY": 1.0,
        "DirectionID": 0,
    })
    return pd.DataFrame(rows)

def build_oam_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.insert(0, "OAM", [f"SM{i:03d}" for i in range(1, len(out) + 1)])
    out = out[["OAM"] + CANON_X + [CANON_Y]].copy()

    rename = {CANON_X[i-1]: f"A{i}" for i in range(1, 12)}
    rename[CANON_Y] = "A12"
    out = out.rename(columns=rename)
    return out

def build_rank_matrix(oam: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rank = pd.DataFrame({"OAM": oam["OAM"]})
    for i in range(1, 12):
        aid = f"A{i}"
        direction_id = int(meta.loc[meta["AttributeID"] == aid, "DirectionID"].iloc[0])
        ascending = True if direction_id == 1 else False
        r = pd.to_numeric(oam[aid], errors="coerce").rank(method="min", ascending=ascending)
        rank[aid] = r.astype("Int64")
    rank["A12"] = pd.to_numeric(oam["A12"], errors="coerce")
    return rank