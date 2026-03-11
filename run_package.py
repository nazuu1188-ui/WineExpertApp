import io
import json
import zipfile
import pandas as pd

def build_run_zip(run_meta: dict, tables: dict[str, pd.DataFrame]) -> bytes:
    """
    Create a 'run package' zip in memory:
      - run_meta.json
      - tables/*.csv
      - README.txt
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("run_meta.json", json.dumps(run_meta, indent=2))

        z.writestr(
            "README.txt",
            "Wine Expert App — Run Package\n"
            "- Raw -> OAM -> Rank -> COCO-STD -> Conclusion\n"
            "- COCO-STD is treated as Similarity space in the thesis.\n"
        )

        for name, df in tables.items():
            if df is None or df.empty:
                continue
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            z.writestr(f"tables/{name}.csv", csv_bytes)

    return buf.getvalue()