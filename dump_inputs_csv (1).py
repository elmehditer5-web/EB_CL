
# dump_inputs_csv.py
import argparse, os, json, datetime, sys
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype, is_integer_dtype, is_float_dtype, is_bool_dtype
)

# --- Insulate against imported modules that call argparse at import time ---
_ORIG_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]  # hide our CLI args from imports that do parse_args()

try:
    from utils.db_related_functions import fetch_data  # returns tuple of 7 DataFrames
except ModuleNotFoundError:
    # fallback to local project layout (files next to this script)
    import importlib.util, pathlib
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from db_related_functions import fetch_data
finally:
    sys.argv = _ORIG_ARGV  # restore real argv for our parser

FILES = [
    ("data_prod_info_eb_zmd", "prod_info_eb_zmd"),
    ("data_croissance",       "croissance"),
    ("data_liaison",          "liaison"),
    ("data_sfp",              "sfp"),
    ("data_fin",              "data_fin"),
    ("data_STG",              "stg"),
    ("data_annexe",           "annexe"),
]

def normalize_and_inspect_types(df: pd.DataFrame):
    df = df.convert_dtypes()
    parse_dates = []
    dtypes_out = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        if is_datetime64_any_dtype(col_dtype):
            parse_dates.append(col)
            dtypes_out[col] = "datetime64[ns]"
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        elif is_integer_dtype(col_dtype):
            dtypes_out[col] = "Int64"
        elif is_float_dtype(col_dtype):
            dtypes_out[col] = "Float64"
        elif is_bool_dtype(col_dtype):
            dtypes_out[col] = "boolean"
        else:
            dtypes_out[col] = "string"
            df[col] = df[col].astype("string")
    return df, dtypes_out, parse_dates

def main():
    ap = argparse.ArgumentParser(description="Export raw SQL inputs to a CSV snapshot (no Parquet)")
    ap.add_argument("--outdir", required=True, help="Directory to write the snapshot into")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"snapshot_csv_{ts}")
    os.makedirs(outdir, exist_ok=True)

    dfs = fetch_data()

    manifest = {
        "created_at": ts,
        "format": "csv",
        "note": "Frozen inputs prior to any calculations",
        "files": []
    }

    for (varname, stem), df in zip(FILES, dfs):
        df_norm, dtypes_out, parse_dates = normalize_and_inspect_types(df.copy())
        csv_path = os.path.join(outdir, f"{stem}.csv")
        df_norm.to_csv(csv_path, index=False, encoding="utf-8")
        manifest["files"].append({
            "name": varname,
            "stem": stem,
            "rows": int(len(df_norm)),
            "csv": os.path.basename(csv_path),
            "schema": {"parse_dates": parse_dates, "dtypes": dtypes_out}
        })

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"CSV snapshot written to: {outdir}")

if __name__ == "__main__":
    main()
