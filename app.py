# app.py
import streamlit as st
from utils import (
    handle_missing_values,
    plot_custom_chart,
    run_regression_diagnostics,
    
    save_csv_to_download
)
import pandas as pd

st.set_page_config(page_title="Regression Diagnostic Tool", layout="wide")
st.title("📉 Regression Assumption Diagnostic Tool")



# Sidebar for upload and options
st.sidebar.header("Upload and Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

# File Upload & Display
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.cleaned_df = df.copy()
    st.sidebar.success("File uploaded successfully!")

    

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values Check")
    df = handle_missing_values(df)

    # Offer cleaned CSV download
    st.download_button("Download Cleaned CSV", save_csv_to_download(df), file_name="cleaned_data.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to begin.")

# Main Diagnostic Interface
if st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.subheader("Feature and Target Selection")
    target_col = st.selectbox("Target Column", numeric_cols)
    feature_cols = st.multiselect("Feature Columns", [col for col in numeric_cols if col != target_col])

    # Sidebar Plot Builder
    with st.sidebar:
        st.header("Custom Plot Builder")
        plot_custom_chart(df)

    # Regression Diagnostics
    if target_col and feature_cols and st.button("Run Regression Diagnostics"):
        run_regression_diagnostics(df, feature_cols, target_col)


