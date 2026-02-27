from itertools import product
from collections.abc import Mapping

import numpy as np
import pandas as pd
import streamlit as st

from doelab import factorial_2k_analyze

from utils.parser_factorial import (
    auto_factor_names,
    grid_to_long,
    long_to_grid,
    map_12_to_pm,
    map_pm_to_12,
    parse_uploaded_table_factorial,
    tuple_2x2_to_long,
    upload_signature,
)


st.title("2^k Factorial Design")
st.caption("Balanced 2-level factorial (2^k) under a CRD")


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


def _make_grid(factors: list[str], r: int) -> pd.DataFrame:
    rows = []
    for combo in product(["-", "+"], repeat=len(factors)):
        combo = combo[::-1]
        row = {f: v for f, v in zip(factors, combo)}
        for j in range(r):
            row[f"Obs {j+1}"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    return df


def _seed_grid_defaults(grid: pd.DataFrame, factors: list[str], r: int) -> pd.DataFrame:
    g = grid.copy()
    obs_cols = [c for c in g.columns if str(c).startswith("Obs ")]
    if not obs_cols:
        return g

    default_grid = {
        ("-", "-"): [28.0, 25.0, 27.0],
        ("+", "-"): [36.0, 32.0, 32.0],
        ("-", "+"): [18.0, 19.0, 23.0],
        ("+", "+"): [31.0, 30.0, 29.0],
    }

    def _is_hi(v: str) -> bool:
        return str(v).strip() == "+"

    if len(factors) == 2:
        a, b = factors[0], factors[1]
        for i in range(len(g)):
            key = (str(g.loc[g.index[i], a]).strip(), str(g.loc[g.index[i], b]).strip())
            vals = default_grid.get(key)
            if not vals:
                continue
            for j, c in enumerate(sorted(obs_cols, key=lambda x: int(str(x).split()[-1]))):
                if j < len(vals) and j < int(r):
                    g.loc[g.index[i], c] = float(vals[j])
                else:
                    g.loc[g.index[i], c] = np.nan
        return g

    # fallback
    base = 50.0
    step = 10.0
    for i in range(len(g)):
        ups = sum(_is_hi(g.loc[g.index[i], f]) for f in factors)
        v0 = base + step * ups
        for j, c in enumerate(sorted(obs_cols, key=lambda x: int(str(x).split()[-1]))):
            g.loc[g.index[i], c] = float(v0 + 2.0 * j)
    return g


def _make_tuple_2x2_table(f1: str, f2: str, r: int) -> pd.DataFrame:
    df = pd.DataFrame({f"{f1}1": ["", ""], f"{f1}2": ["", ""]}, index=[f"{f2}1", f"{f2}2"])

    default_data = {
        ("1", "1"): [28.0, 25.0, 27.0],  # A low, B low
        ("2", "1"): [36.0, 32.0, 32.0],  # A high, B low
        ("1", "2"): [18.0, 19.0, 23.0],  # A low, B high
        ("2", "2"): [31.0, 30.0, 29.0],  # A high, B high
    }

    for b_level, b_idx in [("1", 0), ("2", 1)]:
        for a_level, a_col in [("1", f"{f1}1"), ("2", f"{f1}2")]:
            vals = default_data.get((a_level, b_level), [])
            reps = [str(float(vals[k])) for k in range(min(len(vals), int(r)))]
            if int(r) > len(reps):
                reps.extend([""] * (int(r) - len(reps)))
            df.loc[df.index[b_idx], a_col] = ", ".join([x for x in reps if str(x).strip() != ""])

    return df

def _design_fp(k: int, r: int, factors: list[str]) -> tuple:
    return (int(k), int(r), tuple(factors))


def reconcile_factorial_grid(existing: pd.DataFrame, factors: list[str], r: int) -> pd.DataFrame:
    base = _make_grid(factors, r)

    if existing is None or existing.empty:
        return base

    ex = existing.copy()
    ex_cols = set(ex.columns)

    obs_cols = [f"Obs {i+1}" for i in range(int(r))]
    need_cols = [*factors, *obs_cols]
    for c in need_cols:
        if c not in base.columns:
            base[c] = np.nan

    key_cols = factors
    if not set(key_cols).issubset(ex_cols):
        return base

    ex_key = ex[key_cols].astype(str).apply(lambda s: s.str.strip())
    ex_index = {tuple(row): i for i, row in enumerate(ex_key.to_numpy().tolist())}

    for i in range(len(base)):
        bkey = tuple(str(base.loc[base.index[i], f]).strip() for f in key_cols)
        j = ex_index.get(bkey)
        if j is None:
            continue
        for c in obs_cols:
            if c in ex.columns:
                base.loc[base.index[i], c] = ex.loc[ex.index[j], c]

    return base.reset_index(drop=True)


def _run_labels_from_pm_grid(grid: pd.DataFrame, factors: list[str]) -> pd.Series:
    if grid is None or grid.empty:
        return pd.Series(dtype=str)

    def _label_row(row) -> str:
        terms = []
        for f in factors:
            v = str(row.get(f, "")).strip()
            if v == "+":
                terms.append(str(f).lower())
        return "(1)" if not terms else "".join(terms)

    return grid.apply(_label_row, axis=1).astype(str)


def _init_state() -> None:
    if "ff_k" not in st.session_state:
        st.session_state["ff_k"] = 2
    if "ff_r" not in st.session_state:
        st.session_state["ff_r"] = 3
    if "ff_factors" not in st.session_state:
        st.session_state["ff_factors"] = auto_factor_names(int(st.session_state["ff_k"]))

    if "ff_grid" not in st.session_state:
        f0 = list(st.session_state["ff_factors"])
        r0 = int(st.session_state["ff_r"])
        st.session_state["ff_grid"] = _seed_grid_defaults(_make_grid(f0, r0), f0, r=r0)

    if "ff_table_grid" not in st.session_state:
        f0 = list(st.session_state["ff_factors"])
        r0 = int(st.session_state["ff_r"])
        st.session_state["ff_table_grid"] = (
            _make_tuple_2x2_table(f0[0], f0[1], r=r0) if len(f0) >= 2 else pd.DataFrame()
        )

    if "ff_view_toggle" not in st.session_state:
        st.session_state["ff_view_toggle"] = "Tuple table"

    if "ff_k_widget" not in st.session_state:
        st.session_state["ff_k_widget"] = int(st.session_state["ff_k"])
    if "ff_r_widget" not in st.session_state:
        st.session_state["ff_r_widget"] = int(st.session_state["ff_r"])

    if "ff_pending_widget_update" not in st.session_state:
        st.session_state["ff_pending_widget_update"] = False
    if "ff_pending_k" not in st.session_state:
        st.session_state["ff_pending_k"] = None
    if "ff_pending_r" not in st.session_state:
        st.session_state["ff_pending_r"] = None

    if "ff_last_upload_sig" not in st.session_state:
        st.session_state["ff_last_upload_sig"] = None

    if "ff_design_fp" not in st.session_state:
        st.session_state["ff_design_fp"] = _design_fp(
            int(st.session_state["ff_k"]),
            int(st.session_state["ff_r"]),
            list(st.session_state["ff_factors"]),
        )


def _apply_pending_widget_updates() -> None:
    if not st.session_state.get("ff_pending_widget_update"):
        return

    k = st.session_state.get("ff_pending_k")
    r = st.session_state.get("ff_pending_r")

    if k is not None:
        st.session_state["ff_k_widget"] = int(k)
    if r is not None:
        st.session_state["ff_r_widget"] = int(r)

    st.session_state["ff_pending_widget_update"] = False
    st.session_state["ff_pending_k"] = None
    st.session_state["ff_pending_r"] = None


_init_state()
_apply_pending_widget_updates()

# ----------------- Manual inputs (CRD-style placement) -----------------
k_widget = st.number_input(
    "Number of factors",
    min_value=1,
    max_value=5,
    step=1,
    key="ff_k_widget",
)

r_widget = st.number_input(
    "Replicates per treatment",
    min_value=2,
    max_value=50,
    step=1,
    key="ff_r_widget",
)

k_new = int(k_widget)
r_new = int(r_widget)

factors = list(st.session_state["ff_factors"])
fp_cur = st.session_state.get("ff_design_fp")

fp_new = _design_fp(k_new, r_new, auto_factor_names(k_new))
if fp_cur is None or (k_new != int(st.session_state["ff_k"]) or r_new != int(st.session_state["ff_r"])):
    factors = auto_factor_names(k_new)
    st.session_state["ff_k"] = k_new
    st.session_state["ff_r"] = r_new
    st.session_state["ff_factors"] = factors

    st.session_state["ff_grid"] = reconcile_factorial_grid(st.session_state.get("ff_grid"), factors, r_new)
    if k_new == 2:
        st.session_state["ff_table_grid"] = _make_tuple_2x2_table(factors[0], factors[1], r=r_new)
    else:
        st.session_state["ff_table_grid"] = pd.DataFrame()

    st.session_state["ff_design_fp"] = _design_fp(k_new, r_new, factors)
    st.rerun()

k = int(st.session_state["ff_k"])
r = int(st.session_state["ff_r"])
factors = list(st.session_state["ff_factors"])

# ----------------- View toggle + editor (CRD-style placement) -----------------
if k == 2:
    view = st.segmented_control(
        "View",
        options=["Tuple table", "+/- grid"],
        key="ff_view_toggle",
    )
else:
    view = "+/- grid"

if view == "Tuple table":
    f1, f2 = factors[0], factors[1]
    _tt_index = st.session_state["ff_table_grid"].index

    t_edited = st.data_editor(
        st.session_state["ff_table_grid"],
        use_container_width=True,
        num_rows="fixed",
        key="ff_table_editor",
    )

    if isinstance(t_edited, pd.DataFrame):
        if isinstance(t_edited.index, pd.RangeIndex) and len(t_edited.index) == len(_tt_index):
            t_edited = t_edited.copy()
            t_edited.index = _tt_index
            t_edited.index.name = None

        st.session_state["ff_table_grid"] = t_edited.copy()

        try:
            d12 = tuple_2x2_to_long(st.session_state["ff_table_grid"], f1=f1, f2=f2)
            dpm = map_12_to_pm(d12, [f1, f2])
            g = long_to_grid(dpm[["y", f1, f2]], [f1, f2])

            r_needed = int(st.session_state["ff_r"])
            obs_cols = [c for c in g.columns if str(c).startswith("Obs ")]
            if len(obs_cols) < r_needed:
                for j in range(len(obs_cols), r_needed):
                    g[f"Obs {j+1}"] = np.nan

            obs_cols = [c for c in g.columns if str(c).startswith("Obs ")]
            g = g[[f1, f2, *sorted(obs_cols, key=lambda x: int(str(x).split()[-1]))]]
            st.session_state["ff_grid"] = g.reset_index(drop=True)

        except Exception as e:
            st.error(str(e))

else:
    desired = reconcile_factorial_grid(st.session_state.get("ff_grid"), factors, r)
    desired = desired.copy()
    desired["Label"] = _run_labels_from_pm_grid(desired, factors)

    obs_cols = [c for c in desired.columns if str(c).startswith("Obs ")]
    desired = desired[[*factors, "Label", *sorted(obs_cols, key=lambda x: int(str(x).split()[-1]))]]

    edited = st.data_editor(
        desired,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=[*factors, "Label"],
        key="ff_grid_editor",
    )

    if isinstance(edited, pd.DataFrame):
        st.session_state["ff_grid"] = edited.copy()

        if k == 2:
            f1, f2 = factors[0], factors[1]
            dpm = grid_to_long(st.session_state["ff_grid"], [f1, f2])
            if not dpm.empty:
                d12 = map_pm_to_12(dpm, [f1, f2])
                tab = _make_tuple_2x2_table(f1, f2, r=int(st.session_state["ff_r"]))

                for bi, b_lbl in enumerate(tab.index.tolist()):
                    b_level = b_lbl[len(f2) :].strip() if b_lbl.startswith(f2) else b_lbl
                    for a_col in [f"{f1}1", f"{f1}2"]:
                        a_level = a_col[len(f1) :].strip()
                        ys = d12.loc[
                            (d12[f1].astype(str).str.strip() == a_level)
                            & (d12[f2].astype(str).str.strip() == b_level),
                            "y",
                        ].tolist()
                        if ys:
                            tab.loc[tab.index[bi], a_col] = ", ".join([str(float(x)) for x in ys])

                st.session_state["ff_table_grid"] = tab

# ----------------- Upload beneath editor (CRD-style placement) -----------------
uploaded = st.file_uploader("Upload a CSV/TSV", type=["csv", "tsv", "txt"])
st.caption("Accepted formats: • Long: y + factor columns • Wide: factor columns + Obs columns")

if uploaded is not None:
    try:
        sig = upload_signature(uploaded)

        if sig != st.session_state.get("ff_last_upload_sig"):
            df_long, factor_cols = parse_uploaded_table_factorial(uploaded)

            if df_long.empty or not factor_cols:
                st.error("Uploaded file parsed, but no usable observations were found.")
            else:
                if len(factor_cols) > 5:
                    st.error(f"Upload has {len(factor_cols)} factors; this page is capped at 5.")
                    st.stop()

                factor_cols = [str(c).strip() for c in factor_cols]
                df_long = df_long[["y", *factor_cols]].copy()

                for f in factor_cols:
                    df_long[f] = df_long[f].astype(str).str.strip()
                df_long = map_12_to_pm(df_long, factor_cols)

                grid_from_upload = long_to_grid(df_long, factor_cols)
                if grid_from_upload.empty:
                    st.error("Upload parsed, but could not be reshaped into a grid.")
                    st.stop()

                r_from_upload = max(2, len([c for c in grid_from_upload.columns if str(c).startswith("Obs ")]))
                k_from_upload = int(len(factor_cols))

                st.session_state["ff_grid"] = grid_from_upload.copy()
                st.session_state["ff_k"] = k_from_upload
                st.session_state["ff_r"] = int(r_from_upload)
                st.session_state["ff_factors"] = factor_cols

                if k_from_upload == 2:
                    f1, f2 = factor_cols[0], factor_cols[1]
                    tab = _make_tuple_2x2_table(f1, f2, r=int(r_from_upload))
                    d12 = map_pm_to_12(df_long, [f1, f2])
                    for bi, b_lbl in enumerate(tab.index.tolist()):
                        b_level = b_lbl[len(f2) :].strip() if b_lbl.startswith(f2) else b_lbl
                        for a_col in [f"{f1}1", f"{f1}2"]:
                            a_level = a_col[len(f1) :].strip()
                            ys = d12.loc[
                                (d12[f1].astype(str).str.strip() == a_level)
                                & (d12[f2].astype(str).str.strip() == b_level),
                                "y",
                            ].tolist()
                            if ys:
                                tab.loc[tab.index[bi], a_col] = ", ".join([str(float(x)) for x in ys])
                    st.session_state["ff_table_grid"] = tab
                else:
                    st.session_state["ff_table_grid"] = pd.DataFrame()

                st.session_state["ff_pending_k"] = k_from_upload
                st.session_state["ff_pending_r"] = int(r_from_upload)
                st.session_state["ff_pending_widget_update"] = True

                st.session_state["ff_design_fp"] = _design_fp(
                    st.session_state["ff_k"],
                    st.session_state["ff_r"],
                    list(st.session_state["ff_factors"]),
                )

                st.session_state["ff_last_upload_sig"] = sig
                st.success(f"Loaded {len(df_long)} observation(s) across k={k_from_upload} factor(s).")
                st.rerun()

    except Exception as e:
        st.error(f"Could not parse file: {e}")

# ----------------- Run analysis -----------------
if st.button("Run ANOVA", type="primary"):
    k = int(st.session_state["ff_k"])
    r = int(st.session_state["ff_r"])
    factors = list(st.session_state["ff_factors"])

    df_long = pd.DataFrame()

    if k == 2 and st.session_state.get("ff_view_toggle") == "Tuple table":
        try:
            d12 = tuple_2x2_to_long(st.session_state["ff_table_grid"], f1=factors[0], f2=factors[1])
        except Exception as e:
            st.error(str(e))
            st.stop()
        df_long = map_12_to_pm(d12, factors[:2])
    else:
        df_long = grid_to_long(st.session_state["ff_grid"], factors)

    if df_long.empty:
        st.error("No numeric observations found. Enter at least one value or upload a valid file.")
        st.stop()

    for f in factors:
        lv = sorted(set(df_long[f].astype(str).str.strip().tolist()))
        if len(lv) != 2:
            st.error(f"This page is for 2-level factors only. Factor {f!r} has levels: {lv}.")
            st.stop()

    try:
        res = factorial_2k_analyze(
            df_long,
            response="y",
            factors=factors,
            factor_cols=factors,
            require_complete_cells=True,
            require_replication_per_cell=True,
            min_cell_reps=2,
            allow_unbalanced=False,
            return_tables=("anova", "effects"),
            dump="doe",
            copy=True,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("ANOVA table")
    table = getattr(res, "tables", {}).get("anova")
    if isinstance(table, pd.DataFrame):
        st.dataframe(table.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No ANOVA table was returned.")

    with st.expander("Show contrasts (Fisher–Yates–Kempthorne for 2^k)"):
        eff = getattr(res, "tables", {}).get("effects")

        if not isinstance(eff, pd.DataFrame) or eff.empty:
            st.info("No effects/contrast table was returned.")
        else:
            terms_df = eff[["term"]].copy()
            terms_df["factors"] = terms_df["term"].astype(str).apply(
                lambda t: ", ".join([x for x in str(t).split(":") if x.strip()])
            )
            st.dataframe(terms_df.reset_index(drop=True), use_container_width=True)

            eff_df = eff.copy()
            st.dataframe(eff_df.reset_index(drop=True), use_container_width=True)

    with st.expander("Show intermediate quantities"):
        wtab = workings_to_table(getattr(res, "workings", {}))
        if wtab.empty:
            st.info("No intermediate quantities were returned.")
        else:
            st.dataframe(wtab, use_container_width=True)

    with st.expander("Show data used (long format)"):
        st.dataframe(df_long, use_container_width=True)