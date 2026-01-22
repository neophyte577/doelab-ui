import numpy as np
import pandas as pd
import streamlit as st

from doelab.anova.crd import anova_crd


st.title("CRD — One-way ANOVA")
st.caption("Enter observations by treatment (columns), then run doeLab's CRD ANOVA.")

# Wide format default: columns are treatments, rows are replicates
default_wide = pd.DataFrame(
    {
        "A": [10, 12],
        "B": [13, 15],
        "C": [9, 11],
    }
)

wide = st.data_editor(
    default_wide,
    num_rows="dynamic",
    use_container_width=True,
)

def wide_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = wide_df.copy()

    # Ensure numeric where possible; blanks become NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = (
        df.melt(var_name="treatment", value_name="y")
        .dropna(subset=["y"])
        .reset_index(drop=True)
    )

    long["treatment"] = long["treatment"].astype(str)
    return long


if st.button("Run CRD ANOVA", type="primary"):
    df = wide_to_long(wide)

    if df.empty:
        st.error("No numeric observations found. Enter at least one value.")
        st.stop()

    res = anova_crd(df)

    st.subheader("ANOVA table")
    table = res.table

    # If doeLab returns an index-labeled table, make it display cleanly
    if isinstance(table, pd.DataFrame) and table.index.name is not None:
        table = table.reset_index()

    st.dataframe(table, use_container_width=True)

    with st.expander("Show data used (long format)"):
        st.dataframe(df, use_container_width=True)
