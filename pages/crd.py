import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from doelab import anova_crd


st.title("Completely Randomized Design")
st.caption("Enter observations below, or upload a CSV/TSV to populate the analysis input.")


def grid_to_long(df_grid: pd.DataFrame) -> pd.DataFrame:
    g = df_grid.copy()
    for c in g.columns:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    long = (
        g.stack(dropna=True)
        .reset_index()
        .rename(columns={"Treatment": "treatment", 0: "y"})
    )

    long["treatment"] = long["treatment"].astype(str).str.strip()
    return long.reset_index(drop=True)


def long_to_grid(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    d = df_long.copy()
    if "treatment" not in d.columns or "y" not in d.columns:
        return pd.DataFrame()

    d["treatment"] = d["treatment"].astype(str).str.strip()
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["y"])
    if d.empty:
        return pd.DataFrame()

    groups = []
    order = []
    for t, g in d.groupby("treatment", sort=False):
        order.append(t)
        groups.append(g["y"].to_list())

    k = max(len(vals) for vals in groups)
    cols = [f"Obs {i+1}" for i in range(k)]

    rows = []
    for vals in groups:
        row = list(vals) + [np.nan] * (k - len(vals))
        rows.append(row)

    grid = pd.DataFrame(rows, index=order, columns=cols)
    grid.index.name = "Treatment"
    return grid


def workings_to_table(workings: dict) -> pd.DataFrame:
    rows = []
    for k, v in (workings or {}).items():
        if isinstance(v, (dict, list, tuple, set)):
            v_disp = str(v)
        elif isinstance(v, (float, np.floating)):
            v_disp = float(v)
        elif isinstance(v, (int, np.integer)):
            v_disp = int(v)
        elif isinstance(v, (bool, np.bool_)):
            v_disp = bool(v)
        else:
            v_disp = v
        rows.append({"quantity": k, "value": v_disp})
    return pd.DataFrame(rows)


def _auto_treatment_labels(n: int) -> list[str]:
    labels = []
    i = 0
    while len(labels) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        labels.append(s)
        i += 1
    return labels


def _is_numeric_like(x) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def _row_looks_like_header(row_vals) -> bool:
    toks = [str(x).strip().lower() for x in row_vals]
    header_words = ("treatment", "trt", "group", "obs", "rep", "y", "response")
    return any(any(w in t for w in header_words) for t in toks)


def _read_table_flexible(text: str, header: int | None) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=header)

    if df.shape[1] == 1:
        s = text.strip()
        if "\n" in s:
            sample_lines = [ln for ln in s.splitlines() if ln.strip()][:10]
            if any((" " in ln.strip() or "\t" in ln.strip()) for ln in sample_lines):
                df_ws = pd.read_csv(
                    io.StringIO(text),
                    sep=r"\s+",
                    engine="python",
                    header=header,
                )
                if df_ws.shape[1] > 1:
                    return df_ws

    return df


def _grid_to_long_from_matrix(mat: pd.DataFrame, row_labels: list[str]) -> pd.DataFrame:
    m = mat.copy()
    for c in m.columns:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    m = m.dropna(axis=0, how="all")
    if m.empty:
        return pd.DataFrame(columns=["treatment", "y"])

    row_labels = [str(x).strip() for x in row_labels]
    if len(row_labels) != len(m):
        row_labels = row_labels[: len(m)]

    m.index = pd.Index(row_labels, name="Treatment")
    m.columns = [f"Obs {i+1}" for i in range(m.shape[1])]

    long = (
        m.stack(dropna=True)
        .reset_index()
        .rename(columns={"Treatment": "treatment", 0: "y"})
    )
    long["treatment"] = long["treatment"].astype(str).str.strip()
    return long.reset_index(drop=True)


def parse_uploaded_table(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        txt = content.decode("utf-8")
    except Exception:
        txt = content.decode("latin-1")

    df0 = _read_table_flexible(txt, header=0)

    cols_lower = {str(c).strip().lower(): c for c in df0.columns}
    if "treatment" in cols_lower and "y" in cols_lower:
        tcol = cols_lower["treatment"]
        ycol = cols_lower["y"]

        out = df0[[tcol, ycol]].copy()
        out = out.rename(columns={tcol: "treatment", ycol: "y"})
        out["treatment"] = out["treatment"].astype(str).str.strip()
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
        out = out.dropna(subset=["y"])
        return out.reset_index(drop=True)

    dfN = _read_table_flexible(txt, header=None)

    first_row_vals = dfN.iloc[0].tolist() if not dfN.empty else []
    header_evidence = _row_looks_like_header(first_row_vals)

    df0_colnames = [str(c).strip() for c in df0.columns]
    df0_colnames_all_numeric = bool(df0_colnames) and all(_is_numeric_like(c) for c in df0_colnames)
    df0_row_eaten = (df0.shape[0] + 1 == dfN.shape[0])

    if (df0_colnames_all_numeric and not header_evidence) or (df0_row_eaten and not header_evidence):
        wide = dfN
        has_header = False
    else:
        wide = df0
        has_header = True

    if wide.empty:
        return pd.DataFrame(columns=["treatment", "y"])

    treatment_header_names = {"treatment", "trt", "group"}
    force_label_col = False
    if has_header:
        first_name = str(wide.columns[0]).strip().lower()
        if first_name in treatment_header_names:
            force_label_col = True

    if force_label_col:
        row_labels = wide.iloc[:, 0].astype(str).str.strip().tolist()
        mat = wide.iloc[:, 1:].copy()
    else:
        col0_num = pd.to_numeric(wide.iloc[:, 0], errors="coerce")
        frac_numeric = float(col0_num.notna().mean()) if len(col0_num) else 1.0
        has_row_labels = frac_numeric < 0.5

        if has_row_labels:
            row_labels = wide.iloc[:, 0].astype(str).str.strip().tolist()
            mat = wide.iloc[:, 1:].copy()
        else:
            mat = wide.copy()
            row_labels = _auto_treatment_labels(len(wide))

    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    arr = mat.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).any():
        return pd.DataFrame(columns=["treatment", "y"])

    return _grid_to_long_from_matrix(mat, row_labels)


def reconcile_grid(existing: pd.DataFrame, treatments: list[str], n_obs: int) -> pd.DataFrame:
    cols = [f"Obs {i+1}" for i in range(int(n_obs))]
    out = pd.DataFrame(np.nan, index=treatments, columns=cols)
    out.index.name = "Treatment"

    if existing is None or existing.empty:
        return out

    ex = existing.copy()
    ex.index = ex.index.astype(str).str.strip()

    for t in treatments:
        if t in ex.index:
            for c in cols:
                if c in ex.columns:
                    out.loc[t, c] = ex.loc[t, c]

    return out


def _desired_treatment_labels(existing_grid: pd.DataFrame, n_treatments: int) -> list[str]:
    if existing_grid is None or existing_grid.empty:
        return _auto_treatment_labels(n_treatments)

    base = [str(x).strip() for x in existing_grid.index.tolist()]
    base = [x for x in base if x]

    if len(base) >= n_treatments:
        return base[:n_treatments]

    existing_set = set(base)
    extra = []
    for lbl in _auto_treatment_labels(n_treatments * 2):
        if lbl not in existing_set:
            extra.append(lbl)
        if len(base) + len(extra) >= n_treatments:
            break

    return base + extra


def _init_state() -> None:
    if "crd_n_treatments" not in st.session_state:
        st.session_state["crd_n_treatments"] = 3
    if "crd_n_obs" not in st.session_state:
        st.session_state["crd_n_obs"] = 2

    if "crd_grid" not in st.session_state:
        n_t = int(st.session_state["crd_n_treatments"])
        n_o = int(st.session_state["crd_n_obs"])
        treatments = _auto_treatment_labels(n_t)
        cols = [f"Obs {i+1}" for i in range(n_o)]

        seed = None
        if treatments == ["A", "B", "C"] and n_o == 2:
            seed = pd.DataFrame([[10, 12], [13, 15], [9, 11]], index=treatments, columns=cols)

        grid = seed if seed is not None else pd.DataFrame(np.nan, index=treatments, columns=cols)
        grid.index.name = "Treatment"
        st.session_state["crd_grid"] = grid

    if "crd_n_treatments_widget" not in st.session_state:
        st.session_state["crd_n_treatments_widget"] = int(st.session_state["crd_n_treatments"])
    if "crd_n_obs_widget" not in st.session_state:
        st.session_state["crd_n_obs_widget"] = int(st.session_state["crd_n_obs"])

    if "crd_pending_widget_update" not in st.session_state:
        st.session_state["crd_pending_widget_update"] = False
    if "crd_pending_n_treatments" not in st.session_state:
        st.session_state["crd_pending_n_treatments"] = None
    if "crd_pending_n_obs" not in st.session_state:
        st.session_state["crd_pending_n_obs"] = None

    if "crd_last_upload_sig" not in st.session_state:
        st.session_state["crd_last_upload_sig"] = None


def _apply_pending_widget_updates() -> None:
    if not st.session_state.get("crd_pending_widget_update"):
        return

    n_t = st.session_state.get("crd_pending_n_treatments")
    n_o = st.session_state.get("crd_pending_n_obs")

    if n_t is not None:
        st.session_state["crd_n_treatments_widget"] = int(n_t)
    if n_o is not None:
        st.session_state["crd_n_obs_widget"] = int(n_o)

    st.session_state["crd_pending_widget_update"] = False
    st.session_state["crd_pending_n_treatments"] = None
    st.session_state["crd_pending_n_obs"] = None


def _upload_signature(uploaded) -> str:
    b = uploaded.getvalue()
    h = hashlib.sha256(b).hexdigest()
    return f"{uploaded.name}|{len(b)}|{h}"


_init_state()
_apply_pending_widget_updates()

# ----------------- Manual grid input (persisted) -----------------
n_treatments = st.number_input(
    "Number of treatments",
    min_value=1,
    max_value=200,
    step=1,
    key="crd_n_treatments_widget",
)

n_obs = st.number_input(
    "Observations per treatment",
    min_value=1,
    max_value=50,
    step=1,
    key="crd_n_obs_widget",
)

st.session_state["crd_n_treatments"] = int(n_treatments)
st.session_state["crd_n_obs"] = int(n_obs)

treatments = _desired_treatment_labels(st.session_state["crd_grid"], int(n_treatments))
desired_grid = reconcile_grid(st.session_state["crd_grid"], treatments, int(n_obs))

edited = st.data_editor(
    desired_grid,
    use_container_width=True,
    hide_index=False,
    key="crd_editor",
)

st.session_state["crd_grid"] = edited.copy()

# ----------------- Upload beneath grid (populates grid + persists) -----------------
uploaded = st.file_uploader("Upload a CSV/TSV", type=["csv", "tsv", "txt"])
st.caption(
    "Accepted formats (CSV/TSV/semicolon/whitespace delimited):\n"
    "• Long: columns treatment,y\n"
    "• Wide: numeric matrix (headers/labels optional). If labels are missing, rows are treated as A,B,C... and columns as Obs 1..k."
)

if uploaded is not None:
    try:
        sig = _upload_signature(uploaded)

        if sig != st.session_state.get("crd_last_upload_sig"):
            df_long = parse_uploaded_table(uploaded)
            if df_long.empty:
                st.error("Uploaded file parsed, but no numeric observations were found.")
            else:
                grid_from_upload = long_to_grid(df_long)

                st.session_state["crd_grid"] = grid_from_upload.copy()
                st.session_state["crd_n_treatments"] = int(grid_from_upload.shape[0])
                st.session_state["crd_n_obs"] = int(grid_from_upload.shape[1])

                st.session_state["crd_pending_n_treatments"] = int(grid_from_upload.shape[0])
                st.session_state["crd_pending_n_obs"] = int(grid_from_upload.shape[1])
                st.session_state["crd_pending_widget_update"] = True

                st.session_state["crd_last_upload_sig"] = sig

                st.success(
                    f"Loaded {len(df_long)} observations across "
                    f"{df_long['treatment'].nunique()} treatment(s)."
                )

                st.rerun()

    except Exception as e:
        st.error(f"Could not parse file: {e}")

# ----------------- Run analysis -----------------
if st.button("Run CRD ANOVA", type="primary"):
    df = grid_to_long(st.session_state["crd_grid"])

    if df.empty:
        st.error("No numeric observations found. Enter at least one value or upload a valid file.")
        st.stop()

    res = anova_crd(df, include_workings=True)

    st.subheader("ANOVA table")
    table = res.table
    if isinstance(table, pd.DataFrame):
        table = table.reset_index()
    st.dataframe(table, use_container_width=True)

    with st.expander("Show intermediate quantities"):
        wtab = workings_to_table(getattr(res, "workings", {}))
        if wtab.empty:
            st.info("No intermediate quantities were returned.")
        else:
            st.dataframe(wtab, use_container_width=True)

    with st.expander("Show data used (long format)"):
        st.dataframe(df, use_container_width=True)
