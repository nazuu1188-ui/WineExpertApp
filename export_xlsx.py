import pandas as pd
from io import BytesIO

def to_excel_bytes(sheets: dict[str, pd.DataFrame], round_display: int = 2) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            out = df.copy()
            num_cols = out.select_dtypes(include="number").columns
            out[num_cols] = out[num_cols].round(round_display)
            sheet_name = name[:31]
            out.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
    return buf.getvalue()