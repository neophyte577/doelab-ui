import io
import hashlib
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd
import streamlit as st

try:
    from doelab import rcbd_analyze
except Exception:
    try:
        from doelab.analysis.rcbd import fit_rcbd as rcbd_analyze
    except Exception:
        from doelab.analysis import fit_rcbd as rcbd_analyze


st.title("Randomized Complete Block Design")
st.caption("Enter observations below, or upload a CSV/TSV/XLSX to populate the analysis input.")


def grid_to_long(df_grid: pd.DataFrame) -> pd.DataFrame:
    g = df_grid.copy()
    for c in g.columns:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    long = (
        g.stack(dropna=True)
        .reset_index()
        .rename(columns={"Treatment": "treatment", "Block": "block", 0: "y"})
    )

    long["treatment"] = long["treatment"].astype(str).str.strip()
    long["block"] = long["block"].astype(str).str.strip()
    return long.reset_index(drop=True)


def long_to_grid(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    d = df_long.copy()
    need = {"treatment", "block", "y"}
    if not need.issubset(set(d.columns)):
        return pd.DataFrame()

    d["treatment"] = d["treatment"].astype(str).str.strip()
    d["block"] = d["block"].astype(str).str.strip()
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["treatment", "block", "y"])
    if d.empty:
        return pd.DataFrame()

    counts = d.groupby(["treatment", "block"], sort=False).size()
    if int(counts.max()) > 1:
        raise ValueError(
            "Uploaded RCBD data contain multiple observations in at least one treatment:block cell. "
            "This page currently expects one numeric entry per cell."
        )

    grid = d.pivot(index="treatment", columns="block", values="y")
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    grid.index.name = "Treatment"
    grid.columns.name = "Block"
    return grid


def _coerce_scalar_for_display(v):
    if isinstance(v, (dict, list, tuple, set)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return v


def _flatten_mapping(x, prefix: str = "") -> list[dict]:
    rows = []
    if x is None:
        return rows

    if isinstance(x, Mapping):
        for k, v in x.items():
            key = f"{prefix}{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                rows.extend(_flatten_mapping(v, prefix=f"{key}."))
            else:
                rows.append({"quantity": key, "value": _coerce_scalar_for_display(v)})
        return rows

    rows.append({"quantity": prefix.rstrip("."), "value": _coerce_scalar_for_display(x)})
    return rows


def workings_to_table(workings) -> pd.DataFrame:
    if not isinstance(workings, Mapping) or not workings:
        return pd.DataFrame(columns=["quantity", "value"])

    doe = workings.get("doe", None)
    if isinstance(doe, Mapping):
        rows = _flatten_mapping(doe, prefix="doe.")
    else:
        rows = _flatten_mapping(workings, prefix="")

    return pd.DataFrame(rows, columns=["quantity", "value"])


def _auto_labels(n: int, prefix: str) -> list[str]:
    if prefix == "Treatment":
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
    return [f"B{i}" for i in range(1, n + 1)]


def _is_numeric_like(x) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def _row_looks_like_header(row_vals) -> bool:
    toks = [str(x).strip().lower() for x in row_vals]
    header_words = ("treatment", "trt", "group", "block", "blk", "y", "response")
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


def _is_excel_upload(uploaded_file) -> bool:
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    return suffix in {".xlsx"}


def _read_uploaded_table(uploaded_file, header: int | None) -> pd.DataFrame:
    if _is_excel_upload(uploaded_file):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, header=header)

    content = uploaded_file.getvalue()
    try:
        txt = content.decode("utf-8")
    except Exception:
        txt = content.decode("latin-1")
    return _read_table_flexible(txt, header)


def _grid_to_long_from_matrix(mat: pd.DataFrame, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    m = mat.copy()
    for c in m.columns:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    m = m.dropna(axis=0, how="all")
    m = m.dropna(axis=1, how="all")
    if m.empty:
        return pd.DataFrame(columns=["treatment", "block", "y"])

    row_labels = [str(x).strip() for x in row_labels][: len(m)]
    col_labels = [str(x).strip() for x in col_labels][: m.shape[1]]

    m.index = pd.Index(row_labels, name="Treatment")
    m.columns = pd.Index(col_labels, name="Block")

    long = (
        m.stack(dropna=True)
        .reset_index()
        .rename(columns={"Treatment": "treatment", "Block": "block", 0: "y"})
    )
    long["treatment"] = long["treatment"].astype(str).str.strip()
    long["block"] = long["block"].astype(str).str.strip()
    return long.reset_index(drop=True)


def parse_uploaded_table(uploaded_file) -> pd.DataFrame:
    df0 = _read_uploaded_table(uploaded_file, header=0)
    cols_lower = {str(c).strip().lower(): c for c in df0.columns}
    if {"treatment", "block", "y"}.issubset(cols_lower):
        out = df0[[cols_lower["treatment"], cols_lower["block"], cols_lower["y"]]].copy()
        out = out.rename(
            columns={
                cols_lower["treatment"]: "treatment",
                cols_lower["block"]: "block",
                cols_lower["y"]: "y",
            }
        )
        out["treatment"] = out["treatment"].astype(str).str.strip()
        out["block"] = out["block"].astype(str).str.strip()
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
        out = out.dropna(subset=["y"])
        return out.reset_index(drop=True)

    dfN = _read_uploaded_table(uploaded_file, header=None)
    first_row_vals = dfN.iloc[0].tolist() if not dfN.empty else []
    header_evidence = _row_looks_like_header(first_row_vals)

    df0_colnames = [str(c).strip() for c in df0.columns]
    df0_colnames_all_numeric = bool(df0_colnames) and all(_is_numeric_like(c) for c in df0_colnames)
    df0_row_eaten = df0.shape[0] + 1 == dfN.shape[0]

    if (df0_colnames_all_numeric and not header_evidence) or (df0_row_eaten and not header_evidence):
        wide = dfN
        has_header = False
    else:
        wide = df0
        has_header = True

    if wide.empty:
        return pd.DataFrame(columns=["treatment", "block", "y"])

    force_label_col = False
    if has_header:
        first_name = str(wide.columns[0]).strip().lower()
        if first_name in {"treatment", "trt", "group"}:
            force_label_col = True

    if force_label_col:
        row_labels = wide.iloc[:, 0].astype(str).str.strip().tolist()
        mat = wide.iloc[:, 1:].copy()
        col_labels = [str(c).strip() for c in wide.columns[1:]]
    else:
        col0_num = pd.to_numeric(wide.iloc[:, 0], errors="coerce")
        frac_numeric = float(col0_num.notna().mean()) if len(col0_num) else 1.0
        has_row_labels = frac_numeric < 0.5

        if has_row_labels:
            row_labels = wide.iloc[:, 0].astype(str).str.strip().tolist()
            mat = wide.iloc[:, 1:].copy()
        else:
            mat = wide.copy()
            row_labels = _auto_labels(len(wide), "Treatment")

        col_labels = [str(c).strip() for c in mat.columns] if has_header else _auto_labels(mat.shape[1], "Block")

    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    arr = mat.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).any():
        return pd.DataFrame(columns=["treatment", "block", "y"])

    return _grid_to_long_from_matrix(mat, row_labels, col_labels)


def reconcile_grid(existing: pd.DataFrame, treatments: list[str], blocks: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=treatments, columns=blocks)
    out.index.name = "Treatment"
    out.columns.name = "Block"

    if existing is None or existing.empty:
        return out

    ex = existing.copy()
    ex.index = ex.index.astype(str).str.strip()
    ex.columns = ex.columns.astype(str).str.strip()

    for t in treatments:
        if t in ex.index:
            for b in blocks:
                if b in ex.columns:
                    out.loc[t, b] = ex.loc[t, b]

    return out


def _desired_labels(existing_grid: pd.DataFrame, n: int, axis: str) -> list[str]:
    if existing_grid is None or existing_grid.empty:
        return _auto_labels(n, axis)

    base = existing_grid.index.tolist() if axis == "Treatment" else existing_grid.columns.tolist()
    base = [str(x).strip() for x in base]
    base = [x for x in base if x]

    if len(base) >= n:
        return base[:n]

    existing_set = set(base)
    extra = []
    for lbl in _auto_labels(n * 2, axis):
        if lbl not in existing_set:
            extra.append(lbl)
        if len(base) + len(extra) >= n:
            break

    return base + extra


def _upload_signature(uploaded) -> str:
    b = uploaded.getvalue()
    h = hashlib.sha256(b).hexdigest()
    return f"{uploaded.name}|{len(b)}|{h}"


def _analysis_mode_to_label(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"fixed", "random"}:
        return m
    return "fixed"


def _rcbd_mode_kwargs(treatment_mode: str, block_mode: str) -> dict:
    return {
        "treatment_role": _analysis_mode_to_label(treatment_mode),
        "block_role": _analysis_mode_to_label(block_mode),
    }


def _rcbd_return_tables(treatment_mode: str, block_mode: str) -> tuple[str, ...]:
    treatment_label = _analysis_mode_to_label(treatment_mode)
    block_label = _analysis_mode_to_label(block_mode)

    if treatment_label == "random" or block_label == "random":
        return ("anova", "variance_components", "random_groups", "random_effects")

    return ("anova",)


def _render_mode_picker(label: str, key: str, default: str = "fixed") -> str:
    options = ["fixed", "random"]

    if key not in st.session_state:
        st.session_state[key] = "random" if str(default).strip().lower() == "random" else "fixed"

    if hasattr(st, "segmented_control"):
        return st.segmented_control(
            label,
            options=options,
            selection_mode="single",
            key=key,
            format_func=lambda x: str(x).title(),
        )

    return st.radio(
        label,
        options=options,
        horizontal=True,
        key=key,
        format_func=lambda x: str(x).title(),
    )


def _init_state() -> None:
    if "rcbd_n_treatments" not in st.session_state:
        st.session_state["rcbd_n_treatments"] = 3
    if "rcbd_n_blocks" not in st.session_state:
        st.session_state["rcbd_n_blocks"] = 3

    if "rcbd_grid" not in st.session_state:
        treatments = _auto_labels(int(st.session_state["rcbd_n_treatments"]), "Treatment")
        blocks = _auto_labels(int(st.session_state["rcbd_n_blocks"]), "Block")
        seed = None
        if treatments == ["A", "B", "C"] and blocks == ["B1", "B2", "B3"]:
            seed = pd.DataFrame(
                [[10, 12, 11], [13, 15, 14], [9, 11, 10]],
                index=treatments,
                columns=blocks,
            )
            seed.index.name = "Treatment"
            seed.columns.name = "Block"
        st.session_state["rcbd_grid"] = seed if seed is not None else pd.DataFrame(np.nan, index=treatments, columns=blocks)
        st.session_state["rcbd_grid"].index.name = "Treatment"
        st.session_state["rcbd_grid"].columns.name = "Block"

    if "rcbd_n_treatments_widget" not in st.session_state:
        st.session_state["rcbd_n_treatments_widget"] = int(st.session_state["rcbd_n_treatments"])
    if "rcbd_n_blocks_widget" not in st.session_state:
        st.session_state["rcbd_n_blocks_widget"] = int(st.session_state["rcbd_n_blocks"])

    if "rcbd_pending_widget_update" not in st.session_state:
        st.session_state["rcbd_pending_widget_update"] = False
    if "rcbd_pending_n_treatments" not in st.session_state:
        st.session_state["rcbd_pending_n_treatments"] = None
    if "rcbd_pending_n_blocks" not in st.session_state:
        st.session_state["rcbd_pending_n_blocks"] = None
    if "rcbd_last_upload_sig" not in st.session_state:
        st.session_state["rcbd_last_upload_sig"] = None

    if "rcbd_treatment_mode" not in st.session_state:
        st.session_state["rcbd_treatment_mode"] = "fixed"
    if "rcbd_block_mode" not in st.session_state:
        st.session_state["rcbd_block_mode"] = "random"


def _apply_pending_widget_updates() -> None:
    if not st.session_state.get("rcbd_pending_widget_update"):
        return

    n_t = st.session_state.get("rcbd_pending_n_treatments")
    n_b = st.session_state.get("rcbd_pending_n_blocks")

    if n_t is not None:
        st.session_state["rcbd_n_treatments_widget"] = int(n_t)
    if n_b is not None:
        st.session_state["rcbd_n_blocks_widget"] = int(n_b)

    st.session_state["rcbd_pending_widget_update"] = False
    st.session_state["rcbd_pending_n_treatments"] = None
    st.session_state["rcbd_pending_n_blocks"] = None


_init_state()
_apply_pending_widget_updates()

mode_col1, mode_col2 = st.columns(2)
with mode_col1:
    treatment_mode = _render_mode_picker(
        "Treatment role",
        key="rcbd_treatment_mode",
        default=st.session_state["rcbd_treatment_mode"],
    )
with mode_col2:
    block_mode = _render_mode_picker(
        "Block role",
        key="rcbd_block_mode",
        default=st.session_state["rcbd_block_mode"],
    )

n_treatments = st.number_input(
    "Number of treatments",
    min_value=1,
    max_value=200,
    step=1,
    key="rcbd_n_treatments_widget",
)

n_blocks = st.number_input(
    "Number of blocks",
    min_value=1,
    max_value=200,
    step=1,
    key="rcbd_n_blocks_widget",
)

st.session_state["rcbd_n_treatments"] = int(n_treatments)
st.session_state["rcbd_n_blocks"] = int(n_blocks)

treatments = _desired_labels(st.session_state["rcbd_grid"], int(n_treatments), "Treatment")
blocks = _desired_labels(st.session_state["rcbd_grid"], int(n_blocks), "Block")
desired_grid = reconcile_grid(st.session_state["rcbd_grid"], treatments, blocks)

edited = st.data_editor(
    desired_grid,
    use_container_width=True,
    hide_index=False,
    key="rcbd_editor",
)

st.session_state["rcbd_grid"] = edited.copy()
st.session_state["rcbd_grid"].index.name = "Treatment"
st.session_state["rcbd_grid"].columns.name = "Block"

uploaded = st.file_uploader("Upload a CSV/TSV/XLSX", type=["csv", "tsv", "txt", "xlsx"])
st.caption(
    "Accepted formats (CSV/TSV/XLSX/semicolon/whitespace delimited):\n"
    "• Long: columns treatment,block,y\n"
    "• Wide: treatment x block matrix (headers/labels optional). If labels are missing, rows are treated as A,B,C... and columns as B1..Bk."
)

if uploaded is not None:
    try:
        sig = _upload_signature(uploaded)
        if sig != st.session_state.get("rcbd_last_upload_sig"):
            df_long = parse_uploaded_table(uploaded)
            if df_long.empty:
                st.error("Uploaded file parsed, but no numeric observations were found.")
            else:
                grid_from_upload = long_to_grid(df_long)
                st.session_state["rcbd_grid"] = grid_from_upload.copy()
                st.session_state["rcbd_grid"].index.name = "Treatment"
                st.session_state["rcbd_grid"].columns.name = "Block"

                st.session_state["rcbd_n_treatments"] = int(grid_from_upload.shape[0])
                st.session_state["rcbd_n_blocks"] = int(grid_from_upload.shape[1])
                st.session_state["rcbd_pending_n_treatments"] = int(grid_from_upload.shape[0])
                st.session_state["rcbd_pending_n_blocks"] = int(grid_from_upload.shape[1])
                st.session_state["rcbd_pending_widget_update"] = True
                st.session_state["rcbd_last_upload_sig"] = sig

                st.success(
                    f"Loaded {len(df_long)} observations across "
                    f"{df_long['treatment'].nunique()} treatment(s) and "
                    f"{df_long['block'].nunique()} block(s)."
                )
                st.rerun()
    except Exception as e:
        st.error(f"Could not parse file: {e}")

if st.button("Run RCBD ANOVA", type="primary"):
    df = grid_to_long(st.session_state["rcbd_grid"])

    if df.empty:
        st.error("No numeric observations found. Enter at least one value or upload a valid file.")
        st.stop()

    mode_kwargs = _rcbd_mode_kwargs(treatment_mode, block_mode)
    return_tables = _rcbd_return_tables(treatment_mode, block_mode)

    try:
        res = rcbd_analyze(
            df,
            return_tables=return_tables,
            dump="doe",
            **mode_kwargs,
        )
    except TypeError:
        try:
            res = rcbd_analyze(
                df,
                response="y",
                treatment="treatment",
                block="block",
                return_tables=return_tables,
                dump="doe",
                **mode_kwargs,
            )
        except Exception as e:
            st.error(str(e))
            st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("ANOVA table")
    table = getattr(res, "tables", {}).get("anova")
    if isinstance(table, pd.DataFrame):
        table = table.reset_index(drop=True)
        st.dataframe(table, use_container_width=True)
    else:
        st.info("No ANOVA table was returned.")

    if "variance_components" in return_tables:
        st.subheader("Variance components")
        table = getattr(res, "tables", {}).get("variance_components")
        if isinstance(table, pd.DataFrame):
            st.dataframe(table.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No variance components table was returned.")

        st.subheader("Random groups")
        table = getattr(res, "tables", {}).get("random_groups")
        if isinstance(table, pd.DataFrame):
            st.dataframe(table.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No random groups table was returned.")

        st.subheader("Random effects")
        table = getattr(res, "tables", {}).get("random_effects")
        if isinstance(table, pd.DataFrame):
            st.dataframe(table.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No random effects table was returned.")

    with st.expander("Show intermediate quantities"):
        wtab = workings_to_table(getattr(res, "workings", {}))
        if wtab.empty:
            st.info("No intermediate quantities were returned.")
        else:
            st.dataframe(wtab, use_container_width=True)

    with st.expander("Show data used (long format)"):
        st.dataframe(df, use_container_width=True)