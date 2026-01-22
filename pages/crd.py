import pandas as pd
import streamlit as st

from doelab.anova.crd import anova_crd


st.set_page_config(page_title="CRD", layout="centered")

st.title("CRD — One-way ANOVA")
st.caption("Enter observations by treatment, then run doeLab's CRD ANOVA.")

default = pd.DataFrame(
    {
        "treatment": ["A", "A", "B", "B", "C", "C"],
        "y": [10, 12, 13, 15, 9, 11],
    }
)

df = st.data_editor(default, num_rows="dynamic", use_container_width=True)

run = st.button("Run CRD ANOVA", type="primary")

if run:
    try:
        res = anova_crd(df)

        st.subheader("ANOVA table")
        st.dataframe(res.table, use_container_width=True)

        # show any extra payload if present
        for attr in ("means", "effects", "residuals", "fitted"):
            if hasattr(res, attr):
                st.subheader(attr.capitalize())
                st.write(getattr(res, attr))

    except Exception as e:
        st.error(str(e))
