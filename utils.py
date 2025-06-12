

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import io

# ----- HANDLE MISSING VALUES -----
def handle_missing_values(df):
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]

    if not missing_info.empty:
        st.warning("Your data contains missing values!")

        null_df = pd.DataFrame({
            "S.No": range(1, len(missing_info) + 1),
            "Column Name": missing_info.index,
            "Missing Count": missing_info.values,
            "Data Type": [df[col].dtype for col in missing_info.index]
        })

        st.dataframe(null_df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

        handling_choice = st.radio("Handle missing values:", ("Drop rows", "Fill (auto: mean/mode)"))

        if st.button("Apply Handling"):
            if handling_choice == "Drop rows":
                df = df.dropna()
                st.success("Dropped rows with missing values.")
            else:
                for col in missing_info.index:
                    if df[col].dtype in ["int64", "float64"]:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode().iloc[0], inplace=True)
                st.success("Filled missing values.")
    else:
        st.success("No missing values detected.")
    return df

# ----- CUSTOM CHARTS -----
def plot_custom_chart(df):
    plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Bar Chart", "Box Plot", "Scatter Plot", "Heatmap", "Pairplot"])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if plot_type == "Histogram":
            col = st.selectbox("Column", numeric_cols)
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")

        elif plot_type == "Bar Chart":
            num_col = st.selectbox("Numeric Column", numeric_cols)
            cat_col = st.selectbox("Categorical Column", cat_cols)
            bar_data = df.groupby(cat_col)[num_col].mean().reset_index()
            sns.barplot(x=cat_col, y=num_col, data=bar_data, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f"Bar Chart of {num_col} by {cat_col}")

        elif plot_type == "Box Plot":
            num_col = st.selectbox("Numeric Column", numeric_cols)
            cat_col = st.selectbox("Categorical Column", cat_cols)
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
            ax.set_title(f"Box Plot of {num_col} by {cat_col}")

        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols)
            sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")

        elif plot_type == "Heatmap":
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")

        elif plot_type == "Pairplot":
            pair_cols = st.multiselect("Select Columns", numeric_cols)
            if len(pair_cols) >= 2:
                st.pyplot(sns.pairplot(df[pair_cols], corner=True).fig)
                return
            else:
                st.warning("Select at least 2 columns.")
                return

        st.pyplot(fig)
        st.download_button("Download Plot", data=fig_to_bytes(fig), file_name="plot.png")

    except Exception as e:
        st.error(f"Error while generating plot: {e}")

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# ----- REGRESSION DIAGNOSTICS -----
def run_regression_diagnostics(df, features, target):
    try:
        X = df[features]
        y = df[target]

        # Combine and drop missing values
        data = pd.concat([X, y], axis=1).dropna()
        if data.empty:
            st.error("No data left after removing missing values. Please handle missing values before running diagnostics.")
            return

        X = data[features]
        y = data[target]

        # Add constant term for OLS
        X_const = add_constant(X)

        model = OLS(y, X_const).fit()
        y_pred = model.predict(X_const)
        residuals = y - y_pred

        # 1. LINEARITY
        st.subheader("1. Linearity - Residual Plot")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals (y - ŷ)")
        ax1.set_title("Residuals vs Predicted Values")
        st.pyplot(fig1)

        st.markdown("""
        ✅ **Insight:**  
        - Residuals should be randomly scattered around 0.  
        - Curves or patterns imply non-linearity.
        """)

        # 1A. HOMOSCEDASTICITY
        st.subheader("1A. Homoscedasticity - Spread of Residuals")
        fig_h, ax_h = plt.subplots()
        ax_h.scatter(y_pred, residuals**2, alpha=0.5, color='green')
        ax_h.set_xlabel("Predicted Values")
        ax_h.set_ylabel("Squared Residuals")
        ax_h.set_title("Homoscedasticity Check")
        st.pyplot(fig_h)

        st.markdown("""
        ✅ **Insight:**  
        - Ideally, spread should be uniform.  
        - A cone shape implies heteroscedasticity.
        """)

        # 2. MULTICOLLINEARITY - VIF
        st.subheader("2. Multicollinearity - VIF")
        try:
            vif_data = []
            for i in range(X.shape[1]):
                vif = variance_inflation_factor(X.values, i)
                vif_data.append(vif)

            vif_df = pd.DataFrame({
                "Feature": X.columns,
                "VIF": vif_data
            })

            st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}))

            st.markdown("""
            ✅ **Insight:**  
            - VIF < 5 is ideal.  
            - VIF > 10 means high multicollinearity.
            """)
        except Exception as e:
            st.warning(f"VIF calculation skipped: {e}")

        # 3. NORMALITY OF RESIDUALS - Q-Q Plot
        st.subheader("3. Normality of Residuals - Q-Q Plot")
        fig2, ax2 = plt.subplots()
        probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot of Residuals")
        st.pyplot(fig2)

        st.markdown("""
        ✅ **Insight:**  
        - Residuals should align along the 45° line.
        """)

        # 4. AUTOCORRELATION - DURBIN-WATSON
        st.subheader("4. Autocorrelation - Durbin-Watson")
        dw = durbin_watson(residuals)
        st.metric("Durbin-Watson Statistic", f"{dw:.2f}")

        st.markdown("""
        ✅ **Insight:**  
        - DW ≈ 2 is ideal.  
        - DW < 1.5 or > 2.5 implies autocorrelation.
        """)

        # 5. MODEL SUMMARY
        st.subheader("5. Regression Model Summary")
        with st.expander("Expand to view model summary"):
            st.code(model.summary().as_text())

    except Exception as e:
        st.error(f"Regression diagnostic failed: {e}")


# ----- SAVE TO CSV -----
def save_csv_to_download(df):
    return df.to_csv(index=False).encode('utf-8')


   

def save_csv_to_download(df):
    return df.to_csv(index=False).encode('utf-8')



