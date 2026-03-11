import hashlib
import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from oam_from_raw import load_raw_table, coerce_wine_schema, build_attribute_meta, build_oam_table, build_rank_matrix
from coco_std_engine import coco_std_backfit, quantize_components, inverse_validation
from export_xlsx import to_excel_bytes
from similarity import cosine_similarity_top_k
from db_logger import init_db, log_run_start, log_run_end, log_export, store_coco_std
from run_package import build_run_zip


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pipeline(raw_df: pd.DataFrame, n_iter: int, step: float):
    meta = build_attribute_meta(raw_df)
    oam = build_oam_table(raw_df)
    rank = build_rank_matrix(oam, meta)

    x_a_cols = [f"A{i}" for i in range(1, 12)]  # A1..A11
    coco = coco_std_backfit(rank, x_a_cols=x_a_cols, y_col="A12", n_iter=n_iter)
    coco_q = quantize_components(coco, step=step, decimals=1)

    val = inverse_validation(quantize_components(coco_std_backfit(rank, x_a_cols=x_a_cols, y_col="A12", n_iter=25), step=0.4, decimals=1),
                             rank, x_a_cols=x_a_cols)

    concl = coco_q.copy()
    concl["validation"] = val.values if hasattr(val, "values") else val

    d_pct = (concl["Delta"] / concl["Tény+0"]).replace([np.inf, -np.inf], np.nan)
    concl["conclusion"] = np.where(
        concl["validation"] == 0,
        "invalid",
        np.where(
            d_pct.abs() < 0.05,
            "neutral",
            np.where(d_pct >= 0.05, "overvalued (should be less valued)", "undervalued (should be more valued)"),
        ),
    )
    return meta, oam, rank, coco_q, concl


st.set_page_config(page_title="Wine Expert App (BProf-ready)", layout="wide")
st.title("Wine Expert App — RAW → OAM → Rank → COCO-STD (=Similarity) → Conclusion")

# DB init
conn = init_db("runs.sqlite")
os.makedirs("runs", exist_ok=True)

tabs = st.tabs(["Run", "Help", "Scalability", "Exports & DB"])

with tabs[1]:
    st.header("Help")
    st.markdown(
        """
**Workflow (one-click pipeline)**  
1) Upload RAW wine dataset (CSV/XLSX).  
2) App enforces the fixed wine schema (11 inputs + quality).  
3) App generates: OAM → Rank matrix → COCO-STD → Conclusion (Δ, Δ%, validation).  
4) Export the run package (ZIP) and/or Excel.

**What is COCO-STD here?**  
In this thesis, COCO-STD output vectors `X(A1)…X(A11)` are treated as the **Similarity space**.  
Objects with similar COCO vectors are interpreted as **similar** under the rule-based evaluation.

**Interpretation**  
- `Becslés` = estimated quality  
- `Tény+0` = actual quality  
- `Delta = actual − estimated`  
- `Delta/Tény` = relative deviation (%)  
- `validation` = inverse-run sign check  
- `conclusion` = neutral / overvalued / undervalued (only if valid)
"""
    )

with tabs[0]:
    uploaded = st.file_uploader(
        "Upload RAW wine dataset (.xlsx or .csv)",
        type=["xlsx", "csv"],
        help="Required columns: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality.",
    )
    if not uploaded:
        st.stop()

    file_bytes = uploaded.getvalue()
    dataset_hash = sha256_bytes(file_bytes)
    dataset_name = uploaded.name

    raw = load_raw_table(uploaded)
    try:
        raw = coerce_wine_schema(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Raw preview (validated)")
    st.dataframe(raw.head(30), use_container_width=True)

    st.sidebar.header("Parameters")
    n_iter = st.sidebar.slider("COCO backfit iterations", 5, 60, 25, 5, help="More iterations can improve fit but increases runtime.")
    step = st.sidebar.selectbox("Component quantization step", [0.1, 0.2, 0.4, 0.5], index=2, help="COCO-style discretization step for X(Ai).")

    run_btn = st.button("Run pipeline (log + export-ready)", type="primary", help="Runs the full RAW→OAM→Rank→COCO-STD→Conclusion pipeline.")

    if run_btn:
        run_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat(timespec="seconds")
        params = {"n_iter": n_iter, "step": step}

        t0 = time.perf_counter()
        log_run_start(conn, run_id, created_at, dataset_name, dataset_hash, n_objects=len(raw), n_attributes=12, params=params)

        try:
            meta, oam, rank, coco_std, concl = pipeline(raw, n_iter=n_iter, step=step)
            runtime = time.perf_counter() - t0
            log_run_end(conn, run_id, runtime_sec=runtime, status="success", error_msg=None)

            # store COCO/Conclusion rows in DB
            store_coco_std(conn, run_id, concl)

            # Save exports to disk
            run_dir = os.path.join("runs", run_id)
            os.makedirs(run_dir, exist_ok=True)

            paths = {}
            for name, df in {
                "raw": raw,
                "attribute_meta": meta,
                "oam": oam,
                "rank": rank,
                "coco_std": coco_std,
                "conclusion": concl,
            }.items():
                p = os.path.join(run_dir, f"{name}.csv")
                df.to_csv(p, index=False)
                paths[name] = p
                log_export(conn, run_id, file_type=name, file_path=p)

            # Build run ZIP
            run_meta = {
                "run_id": run_id,
                "created_at": created_at,
                "dataset_name": dataset_name,
                "dataset_hash": dataset_hash,
                "n_objects": len(raw),
                "n_attributes": 12,
                "params": params,
                "runtime_sec": runtime,
                "status": "success",
            }
            zip_bytes = build_run_zip(run_meta, {
                "raw": raw,
                "attribute_meta": meta,
                "oam": oam,
                "rank": rank,
                "coco_std": coco_std,
                "conclusion": concl,
            })
            zip_path = os.path.join(run_dir, "run_package.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
            log_export(conn, run_id, file_type="zip", file_path=zip_path)

            # Excel export (optional)
            xbytes = to_excel_bytes(
                {"Raw": raw, "AttributeMeta": meta, "OAM": oam, "RankMatrix": rank, "COCO_STD": coco_std, "Conclusion": concl},
                round_display=2,
            )
            xlsx_path = os.path.join(run_dir, "run_export.xlsx")
            with open(xlsx_path, "wb") as f:
                f.write(xbytes)
            log_export(conn, run_id, file_type="xlsx", file_path=xlsx_path)

            st.success(f"Run saved. run_id={run_id} | runtime={runtime:.2f}s")

            st.subheader("Attribute meta")
            st.dataframe(meta, use_container_width=True)

            st.subheader("OAM")
            st.dataframe(oam.head(50), use_container_width=True)

            st.subheader("Rank matrix")
            st.dataframe(rank.head(50), use_container_width=True)

            st.subheader("COCO-STD (=Similarity) output")
            st.dataframe(coco_std.head(50), use_container_width=True)

            st.subheader("Conclusion")
            st.dataframe(concl.head(50), use_container_width=True)

            st.markdown("### Derived similarity list (still inside COCO space)")
            feat_cols = [c for c in coco_std.columns if c.startswith("X(A")]
            target = st.selectbox("Target OAM", coco_std["OAM"].tolist(), index=0)
            k = st.slider("Top-K similar", 3, 25, 10)
            topk = cosine_similarity_top_k(
                df=coco_std.rename(columns={"OAM": "ID"}),
                id_col="ID",
                target_id=target,
                feature_cols=feat_cols,
                k=k,
            )
            st.dataframe(topk[["ID", "Similarity", "Becslés", "Tény+0", "Delta", "Delta/Tény"]], use_container_width=True)

            st.download_button(
                "Download run package ZIP",
                data=zip_bytes,
                file_name=f"run_package_{run_id}.zip",
                mime="application/zip",
                help="ZIP contains run_meta.json and all tables as CSV."
            )
            st.download_button(
                "Download Excel export",
                data=xbytes,
                file_name=f"run_export_{run_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            runtime = time.perf_counter() - t0
            log_run_end(conn, run_id, runtime_sec=runtime, status="fail", error_msg=str(e))
            st.error(f"Run failed: {e}")

with tabs[2]:
    st.header("Scalability (batch runs)")
    st.caption("This generates many runs and writes them into runs.sqlite + runs/<run_id>/ exports.")

    uploaded2 = st.file_uploader("Upload RAW dataset again for batch runs", type=["xlsx", "csv"], key="batch_upl")
    if not uploaded2:
        st.stop()

    raw2 = load_raw_table(uploaded2)
    raw2 = coerce_wine_schema(raw2)

    sizes = st.multiselect("Dataset sizes to test", [100, 250, 500, 1000, 1500], default=[100, 250, 500])
    runs_per_size = st.number_input("Runs per size", min_value=1, max_value=50, value=10, step=1)
    n_iter_b = st.slider("Batch iterations", 5, 60, 25, 5, key="batch_iter")
    step_b = st.selectbox("Batch quantization step", [0.1, 0.2, 0.4, 0.5], index=2, key="batch_step")

    start_batch = st.button("Run batch", type="primary")

    if start_batch:
        results = []
        prog = st.progress(0)
        total = len(sizes) * int(runs_per_size)
        done = 0

        for size in sizes:
            if size > len(raw2):
                continue
            for r_i in range(int(runs_per_size)):
                # deterministic sampling per run
                subset = raw2.sample(n=size, random_state=1000 + size * 10 + r_i).reset_index(drop=True)

                run_id = str(uuid.uuid4())
                created_at = datetime.now().isoformat(timespec="seconds")
                params = {"batch": True, "size": size, "rep": r_i, "n_iter": n_iter_b, "step": step_b}

                t0 = time.perf_counter()
                log_run_start(conn, run_id, created_at, uploaded2.name, sha256_bytes(uploaded2.getvalue()),
                              n_objects=len(subset), n_attributes=12, params=params)
                try:
                    meta, oam, rank, coco_std, concl = pipeline(subset, n_iter=int(n_iter_b), step=float(step_b))
                    runtime = time.perf_counter() - t0
                    log_run_end(conn, run_id, runtime, "success", None)
                    store_coco_std(conn, run_id, concl)

                    results.append({"size": size, "rep": r_i, "runtime_sec": runtime, "status": "success", "run_id": run_id})
                except Exception as e:
                    runtime = time.perf_counter() - t0
                    log_run_end(conn, run_id, runtime, "fail", str(e))
                    results.append({"size": size, "rep": r_i, "runtime_sec": runtime, "status": "fail", "run_id": run_id})

                done += 1
                prog.progress(min(1.0, done / total))

        df_res = pd.DataFrame(results)
        st.subheader("Batch run results")
        st.dataframe(df_res, use_container_width=True)

        if not df_res.empty:
            agg = df_res.groupby("size", as_index=False).agg(
                runs=("run_id", "count"),
                mean_runtime=("runtime_sec", "mean"),
                max_runtime=("runtime_sec", "max"),
                fails=("status", lambda s: int((s == "fail").sum())),
            )
            st.subheader("Runtime summary")
            st.dataframe(agg, use_container_width=True)
            st.line_chart(agg.set_index("size")["mean_runtime"])

with tabs[3]:
    st.header("Exports & DB proof")
    st.markdown(
        """
**Where results are stored**
- SQLite DB: `runs.sqlite`
- Files per run: `runs/<run_id>/...`

For thesis screenshots:
- Show `runs.sqlite` opened in a SQLite viewer (runs table filled).
- Show `runs/` folder with many run_id subfolders.
"""
    )