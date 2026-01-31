import io
import hashlib
from typing import Optional

import numpy as np
import pandas as pd


def upload_signature(uploaded) -> str:
    b = uploaded.getvalue()
    h = hashlib.sha256(b).hexdigest()
    return f"{uploaded.name}|{len(b)}|{h}"


def auto_factor_names(k: int) -> list[str]:
    names = []
    i = 0
    while len(names) < k:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        names.append(s)
        i += 1
    return names


def read_table_flexible(text: str, header: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=header)

    if df.shape[1] == 1:
        s = text.strip()
        if "\n" in s:
            sample_lines = [ln for ln in s.splitlines() if ln.strip()][:10]
            if any((" " in ln.strip() or "\t" in ln.strip()) for ln in sample_lines):
                df_ws = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", header=header)
                if df_ws.shape[1] > 1:
                    return df_ws

    return df


def map_12_to_pm(d: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    out = d.copy()
    for f in factors:
        out[f] = out[f].astype(str).str.strip().map({"1": "-", "2": "+"}).fillna(out[f])
    return out


def map_pm_to_12(d: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    out = d.copy()
    for f in factors:
        out[f] = out[f].astype(str).str.strip().map({"-": "1", "+": "2"}).fillna(out[f])
    return out


def normalize_two_level_obvious(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    tok = s.str.lower()

    uniq = [u for u in pd.unique(tok) if u not in ("", "nan", "none")]
    if len(uniq) != 2:
        return series

    u = set(uniq)

    if u == {"-", "+"}:
        return tok.map({"-": "-", "+": "+"}).astype(str)
    if u == {"1", "2"}:
        return tok.map({"1": "-", "2": "+"}).astype(str)
    if u == {"0", "1"}:
        return tok.map({"0": "-", "1": "+"}).astype(str)
    if u == {"-1", "1"}:
        return tok.map({"-1": "-", "1": "+"}).astype(str)
    if u == {"low", "high"}:
        return tok.map({"low": "-", "high": "+"}).astype(str)
    if u == {"l", "h"}:
        return tok.map({"l": "-", "h": "+"}).astype(str)

    return series


def grid_to_long(grid: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    if grid is None or grid.empty:
        return pd.DataFrame(columns=["y", *factors])

    g = grid.copy()
    obs_cols = [c for c in g.columns if str(c).startswith("Obs ")]
    if not obs_cols:
        return pd.DataFrame(columns=["y", *factors])

    for c in obs_cols:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    for f in factors:
        g[f] = g[f].astype(str).str.strip()

    long = (
        g.melt(id_vars=factors, value_vars=obs_cols, var_name="rep", value_name="y")
        .drop(columns=["rep"])
    )
    long["y"] = pd.to_numeric(long["y"], errors="coerce")
    long = long.dropna(subset=["y", *factors])
    return long.reset_index(drop=True)


def long_to_grid(df_long: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    d = df_long.copy()
    need = {"y", *factors}
    if not need.issubset(set(d.columns)):
        return pd.DataFrame()

    for f in factors:
        d[f] = d[f].astype(str).str.strip()
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["y", *factors])
    if d.empty:
        return pd.DataFrame()

    groups = []
    order = []
    for key, g in d.groupby(factors, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        order.append(tuple(str(x).strip() for x in key))
        groups.append(g["y"].to_list())

    r_max = max(len(v) for v in groups)
    obs_cols = [f"Obs {i+1}" for i in range(r_max)]

    rows = []
    for key, vals in zip(order, groups):
        row = {factors[i]: key[i] for i in range(len(factors))}
        pad = list(vals) + [np.nan] * (r_max - len(vals))
        for c, v in zip(obs_cols, pad):
            row[c] = v
        rows.append(row)

    out = pd.DataFrame(rows)
    out.index = pd.RangeIndex(start=0, stop=len(out), step=1)
    return out.reset_index(drop=True)


def parse_cell_reps(x: object) -> list[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    s = s.strip("()[]{}")
    s = s.replace(";", ",")
    parts = [p for p in s.replace("\t", " ").replace("\n", " ").split(",") if p.strip()]
    if len(parts) == 1 and " " in parts[0].strip():
        parts = [p for p in parts[0].split(" ") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p.strip()))
        except Exception:
            pass
    return out


def tuple_2x2_to_long(table_grid: pd.DataFrame, f1: str, f2: str) -> pd.DataFrame:
    if table_grid is None or table_grid.empty:
        return pd.DataFrame(columns=["y", f1, f2])

    t = table_grid.copy()

    need_cols = {f"{f1}1", f"{f1}2"}
    if not need_cols.issubset(set(t.columns)) or len(t.index) != 2:
        return pd.DataFrame(columns=["y", f1, f2])

    rows = []
    cell_ns = []
    filled = 0

    for i in range(len(t)):
        b_lbl = str(t.index[i]).strip()
        if not b_lbl:
            continue
        b_level = b_lbl[len(f2) :].strip() if b_lbl.startswith(f2) else b_lbl

        for a_col in [f"{f1}1", f"{f1}2"]:
            a_level = a_col[len(f1) :].strip()
            reps = parse_cell_reps(t.loc[t.index[i], a_col])
            if reps:
                filled += 1
                cell_ns.append(len(reps))
            for y in reps:
                rows.append({"y": y, f1: a_level, f2: b_level})

    if not rows:
        return pd.DataFrame(columns=["y", f1, f2])

    if filled != 4:
        raise ValueError(f"Table entry is missing data in one or more cells. Filled cells: {filled}/4.")
    if len(set(cell_ns)) != 1:
        raise ValueError(
            "Table entry must be balanced: each cell must contain the same number of replicates. "
            f"Observed replicate counts: {sorted(set(cell_ns))}"
        )
    if cell_ns[0] < 2:
        raise ValueError("Table entry requires replication r>=2 in every cell.")

    df_long = pd.DataFrame(rows)
    df_long["y"] = pd.to_numeric(df_long["y"], errors="coerce")
    df_long = df_long.dropna(subset=["y", f1, f2]).reset_index(drop=True)
    return df_long


def parse_uploaded_table_factorial(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    content = uploaded_file.getvalue()
    try:
        txt = content.decode("utf-8")
    except Exception:
        txt = content.decode("latin-1")

    def _clean_levels(s: pd.Series) -> pd.Series:
        v = s.astype(str).str.strip()
        v = v.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
        return v

    def _two_levelish(s: pd.Series) -> bool:
        v = _clean_levels(s).dropna()
        if v.empty:
            return False
        v = normalize_two_level_obvious(v).astype(str).str.strip()
        return len(set(v.tolist())) == 2

    def _numericish(s: pd.Series) -> bool:
        x = pd.to_numeric(s, errors="coerce")
        ok = int(x.notna().sum())
        return ok >= max(2, int(0.6 * len(s)))

    def _first_nonempty_line(text: str) -> str:
        for ln in text.splitlines():
            if ln.strip():
                return ln.strip()
        return ""

    def _tokenize_guess(line: str) -> list[str]:
        if not line:
            return []
        if "," in line:
            toks = [t.strip() for t in line.split(",")]
            return [t for t in toks if t != ""]
        if "\t" in line:
            toks = [t.strip() for t in line.split("\t")]
            return [t for t in toks if t != ""]
        toks = [t.strip() for t in line.replace("\t", " ").split()]
        return [t for t in toks if t != ""]

    def _looks_headerless_by_first_row(text: str) -> bool:
        line = _first_nonempty_line(text)
        toks = _tokenize_guess(line)
        if not toks:
            return False

        norm = [t.strip().lower() for t in toks]
        level_toks = {"+", "-", "high", "low", "h", "l", "1", "2", "0", "-1"}

        if any(t in level_toks for t in norm):
            return True

        numeric_ct = 0
        for t in norm:
            try:
                float(t)
                numeric_ct += 1
            except Exception:
                pass
        return numeric_ct >= max(1, int(0.5 * len(norm)))

    def _reject_numericish_headerless_pattern(dfN: pd.DataFrame) -> None:
        if dfN is None or dfN.empty:
            return

        r0 = dfN.iloc[0].tolist()
        toks = []
        for x in r0:
            s = str(x).strip().lower()
            if not s or s in ("nan", "none"):
                continue
            toks.append(s)

        if not toks:
            return

        # STRICT: if the first row contains ANY numeric-coded level labels, throw error
        
        numeric_level_tokens = {"-1", "0", "1", "2"}
        if any(t in numeric_level_tokens for t in toks):
            raise ValueError(
                "Headerless file appears to start with numeric-coded factor levels "
                "(e.g., -1/1, 0/1, 1/2). This format is not accepted here. "
                "Use +/- (or low/high, h/l) for factor levels, or provide a headered long table."
            )


    def _best_two_level_run(dfN: pd.DataFrame) -> tuple[int, int]:
        if dfN is None or dfN.empty or dfN.shape[1] < 1:
            return (-1, -1)

        flags = []
        for c in dfN.columns:
            try:
                flags.append(bool(_two_levelish(dfN[c])))
            except Exception:
                flags.append(False)

        best = (-1, -1)
        best_len = 0
        i = 0
        while i < len(flags):
            if not flags[i]:
                i += 1
                continue
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best = (i, j)
            i = j

        return best

    def _parse_headerless_wide_by_run(dfN: pd.DataFrame) -> tuple[pd.DataFrame, list[str]] | None:
        i0, i1 = _best_two_level_run(dfN)
        if i0 < 0 or i1 <= i0:
            return None

        cols = list(dfN.columns)
        factor_pos = cols[i0:i1]
        left = cols[:i0]
        right = cols[i1:]

        if not left and not right:
            return None

        left_num = [c for c in left if _numericish(dfN[c])]
        right_num = [c for c in right if _numericish(dfN[c])]

        if left_num and right_num:
            raise ValueError(
                "Headerless +/- level grid detected, but numeric observations were found on BOTH sides of the level grid. "
                "Place observations exclusively to the LEFT or exclusively to the RIGHT of the level grid."
            )

        obs_side = None
        obs_pos = None
        if left_num:
            obs_side = "left"
            obs_pos = left
        elif right_num:
            obs_side = "right"
            obs_pos = right
        else:
            return None

        if obs_pos is None or not obs_pos:
            return None

        k = int(len(factor_pos))
        factor_names = auto_factor_names(k)

        grid = dfN[[*factor_pos, *obs_pos]].copy()
        grid = grid.rename(columns={factor_pos[i]: factor_names[i] for i in range(k)})

        obs_names = [f"Obs {i+1}" for i in range(len(obs_pos))]
        grid = grid.rename(columns={obs_pos[i]: obs_names[i] for i in range(len(obs_pos))})

        for f in factor_names:
            grid[f] = _clean_levels(grid[f])
        grid = map_12_to_pm(grid, factor_names)

        for c in obs_names:
            grid[c] = pd.to_numeric(grid[c], errors="coerce")

        long = grid_to_long(grid, factor_names)
        return long, factor_names

   
    headerless_hint = _looks_headerless_by_first_row(txt)

    if not headerless_hint:
        df0 = read_table_flexible(txt, header=0)
        df0.columns = [str(c).strip() for c in df0.columns]
        cols_lower = {str(c).strip().lower(): c for c in df0.columns}

        if "y" in cols_lower:
            ycol = cols_lower["y"]
            factor_cols = [c for c in df0.columns if c != ycol]
            if not factor_cols:
                return pd.DataFrame(), []
            out = df0[[ycol, *factor_cols]].rename(columns={ycol: "y"}).copy()
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
            for f in factor_cols:
                out[f] = normalize_two_level_obvious(out[f]).astype(str).str.strip()
            out = out.dropna(subset=["y", *factor_cols]).reset_index(drop=True)
            return out, factor_cols

        num_cols = [c for c in df0.columns if _numericish(df0[c])]
        two_cols = [c for c in df0.columns if _two_levelish(df0[c])]

        if len(num_cols) == 1 and two_cols:
            ycol = num_cols[0]
            factor_cols = [c for c in df0.columns if c != ycol and _two_levelish(df0[c])]
            if factor_cols:
                out = df0[[ycol, *factor_cols]].rename(columns={ycol: "y"}).copy()
                out["y"] = pd.to_numeric(out["y"], errors="coerce")
                for f in factor_cols:
                    out[f] = normalize_two_level_obvious(out[f]).astype(str).str.strip()
                out = out.dropna(subset=["y", *factor_cols]).reset_index(drop=True)
                return out, factor_cols

        if two_cols:
            factor_pos = []
            for c in df0.columns:
                if _two_levelish(df0[c]):
                    factor_pos.append(c)
                else:
                    break

            if factor_pos:
                rest = [c for c in df0.columns if c not in factor_pos]
                obs_pos = [c for c in rest if _numericish(df0[c])]

                if obs_pos:
                    grid = df0[[*factor_pos, *obs_pos]].copy()

                    for f in factor_pos:
                        grid[f] = normalize_two_level_obvious(grid[f]).astype(str).str.strip()

                    obs_names = [f"Obs {i+1}" for i in range(len(obs_pos))]
                    grid = grid.rename(columns={obs_pos[i]: obs_names[i] for i in range(len(obs_pos))})

                    for c in obs_names:
                        grid[c] = pd.to_numeric(grid[c], errors="coerce")

                    long = grid_to_long(grid, list(factor_pos))
                    return long, [str(c).strip() for c in factor_pos]

    dfN = read_table_flexible(txt, header=None)
    if dfN is None or dfN.empty or dfN.shape[1] < 2:
        return pd.DataFrame(), []

    _reject_numericish_headerless_pattern(dfN)

    inferred = _parse_headerless_wide_by_run(dfN)
    if inferred is not None:
        return inferred

    numeric_cols = [c for c in dfN.columns if _numericish(dfN[c])]
    if numeric_cols:
        ycol = numeric_cols[0]
        cand = [c for c in dfN.columns if c != ycol]
        factor_pos = [c for c in cand if _two_levelish(dfN[c])]
        if factor_pos:
            k = int(len(factor_pos))
            factor_names = auto_factor_names(k)
            out = dfN[[ycol, *factor_pos]].copy()
            out = out.rename(columns={ycol: "y", **{factor_pos[i]: factor_names[i] for i in range(k)}})
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
            for f in factor_names:
                out[f] = _clean_levels(out[f])
            out = map_12_to_pm(out, factor_names)
            out = out.dropna(subset=["y", *factor_names]).reset_index(drop=True)
            return out, factor_names

    two_level_cols = [c for c in dfN.columns if _two_levelish(dfN[c])]
    if not two_level_cols:
        return pd.DataFrame(), []

    factor_pos = []
    for c in dfN.columns:
        if _two_levelish(dfN[c]):
            factor_pos.append(c)
        else:
            break

    if not factor_pos:
        return pd.DataFrame(), []

    obs_pos = [c for c in dfN.columns if c not in factor_pos]
    if not obs_pos:
        return pd.DataFrame(), []

    k = int(len(factor_pos))
    factor_names = auto_factor_names(k)

    grid = dfN[[*factor_pos, *obs_pos]].copy()
    grid = grid.rename(columns={factor_pos[i]: factor_names[i] for i in range(k)})

    obs_names = [f"Obs {i+1}" for i in range(len(obs_pos))]
    grid = grid.rename(columns={obs_pos[i]: obs_names[i] for i in range(len(obs_pos))})

    for f in factor_names:
        grid[f] = _clean_levels(grid[f])
    grid = map_12_to_pm(grid, factor_names)

    for c in obs_names:
        grid[c] = pd.to_numeric(grid[c], errors="coerce")

    long = grid_to_long(grid, factor_names)
    return long, factor_names
