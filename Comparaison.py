import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
# --------- Header & reading helpers ---------
def _clean_header(name: str) -> str:
   if name is None:
       return ""
   # remove BOM + non-breaking space + trim + lowercase
   s = str(name).replace("\ufeff", "").replace("\xa0", " ").strip().lower()
   # collapse whitespace to single space
   s = re.sub(r"\s+", " ", s)
   # unify separators: spaces/hyphens -> underscore
   s = s.replace("-", "_").replace(" ", "_")
   return s
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
   out = df.copy()
   out.columns = [_clean_header(c) for c in out.columns]
   # trim string cells
   for col in out.select_dtypes(include=["object", "string"]).columns:
       out[col] = out[col].astype("string").str.replace("\ufeff", "", regex=False)\
                                          .str.replace("\xa0", " ", regex=False)\
                                          .str.strip()
   return out
def read_csv_resilient(path: str, sep_hint: str | None):
   """Try reading with given sep or fallback list when file collapses to 1 column."""
   path = Path(path)
   if not path.exists():
       raise FileNotFoundError(f"File not found: {path}")
   seps = [sep_hint] if sep_hint else [None, ";", ",", "\t", "|"]
   last_err = None
   for sep in seps:
       try:
           df = pd.read_csv(path, sep=sep, engine="python")
           # If only 1 column and line looks like delimited text, retry with other sep
           if df.shape[1] == 1 and sep is None:
               text = path.read_text(errors="ignore")
               if ";" in text:
                   continue
           return df
       except Exception as e:
           last_err = e
           continue
   raise RuntimeError(f"Failed to read {path} with any separator. Last error: {last_err}")
def dedup_on_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
   if df.duplicated(subset=keys).any():
       df = df.sort_values(keys).drop_duplicates(subset=keys, keep="first")
   return df
def align_on_keys(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
   old_ix = old.set_index(keys, drop=False).sort_index()
   new_ix = new.set_index(keys, drop=False).sort_index()
   common = old_ix.index.intersection(new_ix.index)
   return old_ix.loc[common].copy(), new_ix.loc[common].copy()
def find_key_variant(df: pd.DataFrame, wanted: str) -> str | None:
   """Return an existing column name that matches 'wanted' after cleaning."""
   wanted_clean = _clean_header(wanted)
   for c in df.columns:
       if _clean_header(c) == wanted_clean:
           return c
   # also try loose matching: remove underscores
   wanted_loose = wanted_clean.replace("_", "")
   for c in df.columns:
       if _clean_header(c).replace("_", "") == wanted_loose:
           return c
   return None
# --------- Compare helpers ---------
def compare_numeric(a: pd.Series, b: pd.Series, tol_abs: float, tol_rel: float) -> pd.Series:
   both_nan = a.isna() & b.isna()
   a_f = pd.to_numeric(a, errors="coerce")
   b_f = pd.to_numeric(b, errors="coerce")
   diff_abs = (a_f - b_f).abs()
   with np.errstate(divide='ignore', invalid='ignore'):
       diff_rel = diff_abs / b_f.replace(0, np.nan).abs()
   ok = both_nan | (diff_abs <= tol_abs) | (diff_rel <= tol_rel)
   return ~ok.fillna(False)
def compare_columns(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], tol_abs: float, tol_rel: float):
   cols_common = [c for c in old.columns if c in new.columns and c not in keys]
   summary_rows = []
   detail = {}
   for col in cols_common:
       a = old[col]
       b = new[col]
       a_num = pd.to_numeric(a, errors="coerce")
       b_num = pd.to_numeric(b, errors="coerce")
       both_numericish = (~a_num.isna() | a.isna()).all() and (~b_num.isna() | b.isna()).all()
       if both_numericish:
           mis = compare_numeric(a, b, tol_abs, tol_rel)
       else:
           a_str = a.astype("string").fillna("__NA__").str.strip()
           b_str = b.astype("string").fillna("__NA__").str.strip()
           mis = ~(a_str == b_str)
       count = int(mis.sum())
       summary_rows.append({"column": col, "mismatches": count})
       if count:
           cols = {k: old[k] for k in keys}
           detail[col] = pd.DataFrame({**cols, f"old.{col}": a, f"new.{col}": b})[mis]
   summary_df = pd.DataFrame(summary_rows).sort_values("mismatches", ascending=False)
   return summary_df, detail
def write_report(old_raw, new_raw, old_aligned, new_aligned, keys, col_summary, detail, out_dir: Path, tol_abs, tol_rel):
   out_dir.mkdir(parents=True, exist_ok=True)
   old_keys = set(tuple(x) for x in old_raw[keys].astype(str).to_numpy())
   new_keys = set(tuple(x) for x in new_raw[keys].astype(str).to_numpy())
   only_old = sorted(list(old_keys - new_keys))
   only_new = sorted(list(new_keys - old_keys))
   cols_only_old = sorted(set(old_raw.columns) - set(new_raw.columns))
   cols_only_new = sorted(set(new_raw.columns) - set(old_raw.columns))
   summary_txt = [
       "=== COMPARISON SUMMARY ===",
       f"Old rows             : {len(old_raw):,}",
       f"New rows             : {len(new_raw):,}",
       f"Common keys          : {len(old_aligned):,}",
       f"Keys only in OLD     : {len(only_old):,}",
       f"Keys only in NEW     : {len(only_new):,}",
       f"Columns only in OLD  : {len(cols_only_old):,}",
       f"Columns only in NEW  : {len(cols_only_new):,}",
       f"Total mismatches     : {int(col_summary['mismatches'].sum()) if len(col_summary) else 0:,}",
       f"Abs tolerance        : {tol_abs}",
       f"Rel tolerance        : {tol_rel}",
       ""
   ]
   (out_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")
   pd.DataFrame(only_old, columns=keys).to_csv(out_dir / "keys_only_in_old.csv", index=False, encoding="utf-8")
   pd.DataFrame(only_new, columns=keys).to_csv(out_dir / "keys_only_in_new.csv", index=False, encoding="utf-8")
   pd.DataFrame({"columns_only_in_old": cols_only_old}).to_csv(out_dir / "columns_only_in_old.csv", index=False, encoding="utf-8")
   pd.DataFrame({"columns_only_in_new": cols_only_new}).to_csv(out_dir / "columns_only_in_new.csv", index=False, encoding="utf-8")
   col_summary.to_csv(out_dir / "col_mismatch_counts.csv", index=False, encoding="utf-8")
   cap = 5000
   for col, df_mis in detail.items():
       df_out = df_mis if len(df_mis) <= cap else df_mis.iloc[:cap]
       safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in col)[:80]
       df_out.to_csv(out_dir / f"mis_{safe}.csv", index=False, encoding="utf-8")
# --------- Main ---------
def main():
   ap = argparse.ArgumentParser(description="Robust compare of two CSV files (old vs new).")
   ap.add_argument("--old", default="old.csv", help="Path to OLD CSV")
   ap.add_argument("--new", default="new.csv", help="Path to NEW CSV")
   ap.add_argument("--key", nargs="+", default=["id_pm"], help="Key column(s), default: id_pm")
   ap.add_argument("--abs", type=float, default=1e-6, help="Absolute tolerance for numeric compare")
   ap.add_argument("--rel", type=float, default=1e-6, help="Relative tolerance for numeric compare")
   ap.add_argument("--out", default="compare_report", help="Output folder")
   ap.add_argument("--sep-old", default=None, help="Separator for OLD (e.g. ';'). Default: auto")
   ap.add_argument("--sep-new", default=None, help="Separator for NEW (e.g. ';'). Default: auto")
   args = ap.parse_args()
   old_raw = normalize_df(read_csv_resilient(args.old, args.sep_old))
   new_raw = normalize_df(read_csv_resilient(args.new, args.sep_new))
   # Resolve keys against cleaned headers
   resolved_keys = []
   for k in args.key:
       k_found_old = find_key_variant(old_raw, k)
       k_found_new = find_key_variant(new_raw, k)
       if not k_found_old:
           print("Colonnes OLD disponibles :", list(old_raw.columns))
           raise KeyError(f"Key '{k}' not found in OLD file after normalization.")
       if not k_found_new:
           print("Colonnes NEW disponibles :", list(new_raw.columns))
           raise KeyError(f"Key '{k}' not found in NEW file after normalization.")
       if k_found_old != k_found_new:
           # Align column names between frames (rename new to old for consistency)
           new_raw = new_raw.rename(columns={k_found_new: k_found_old})
       resolved_keys.append(k_found_old)
   old_raw = dedup_on_keys(old_raw, resolved_keys)
   new_raw = dedup_on_keys(new_raw, resolved_keys)
   old_al, new_al = align_on_keys(old_raw, new_raw, resolved_keys)
   col_summary, detail = compare_columns(old_al, new_al, resolved_keys, args.abs, args.rel)
   out_dir = Path(args.out)
   write_report(old_raw, new_raw, old_al, new_al, resolved_keys, col_summary, detail, out_dir, args.abs, args.rel)
   print((out_dir / "summary.txt").read_text(encoding="utf-8"))
   print(f"Report files written to: {out_dir.resolve()}")
if __name__ == "__main__":
   main()

#python "C:\Users\ETERRAF\Downloads\Comparaison.py" --old "C:\Users\ETERRAF\Downloads\org1.csv" --new "C:\Users\ETERRAF\Downloads\opt4.csv" --key id_pm --out "C:\Users\ETERRAF\Downloads\compare_report" --abs 1e-4 --rel 1e-4
 
