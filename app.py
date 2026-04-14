import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns

from ydata_profiling import ProfileReport
import sweetviz as sv

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="AI CSV SaaS Dashboard", layout="wide")
st.title("🚀 AI CSV SaaS Dashboard (Pro Level)")

# -----------------------
# AI KEY (OPTIONAL)
# -----------------------
st.sidebar.subheader("🔑 AI Settings (Optional)")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

ai_enabled = api_key is not None and api_key.strip() != ""

if ai_enabled:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        ai_enabled = False
        st.sidebar.error("OpenAI not installed or failed")

# -----------------------
# UPLOAD CSV
# -----------------------
file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None
filtered_df = None
numeric_cols = []

if file:
    try:
        df = pd.read_csv(file)
        st.success("File uploaded successfully!")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        filtered_df = df.copy()

    except Exception:
        st.error("❌ Failed to read CSV file")

# -----------------------
# MAIN APP
# -----------------------
if df is not None:

    # -----------------------
    # DATA PREVIEW
    # -----------------------
    st.subheader("📌 Data Preview")
    st.dataframe(df.head(10))

    st.subheader("📊 Dataset Info")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    # -----------------------
    # SMART FILTERS
    # -----------------------
    st.subheader("🎛️ Smart Filters")

    try:
        if len(numeric_cols) > 0:
            col = st.selectbox("Select filter column", numeric_cols)

            min_val = float(df[col].min())
            max_val = float(df[col].max())

            rng = st.slider("Filter range", min_val, max_val, (min_val, max_val))

            filtered_df = df[
                (df[col] >= rng[0]) &
                (df[col] <= rng[1])
            ]
        else:
            filtered_df = df

    except Exception:
        st.warning("Filter error, showing full data")
        filtered_df = df

    # -----------------------
    # 🤖 AI CHAT (SAFE)
    # -----------------------
    st.subheader("🤖 AI Data Chat")

    if not ai_enabled:
        st.warning("🔒 AI disabled. Add OpenAI API key in sidebar.")

    else:
        question = st.text_input("Ask anything about dataset")

        if question:
            try:
                schema = f"""
You are a data analyst.

Columns:
{list(df.columns)}

Sample:
{df.head(5).to_string()}
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": schema},
                        {"role": "user", "content": question}
                    ]
                )

                st.success("🤖 AI Response")
                st.write(response.choices[0].message.content)

            except Exception as e:
                err = str(e)

                if "401" in err or "invalid_api_key" in err:
                    st.error("🔑 Invalid API Key")
                elif "rate_limit" in err:
                    st.error("⏳ Rate limit reached")
                else:
                    st.error("⚠️ AI error occurred")

    # -----------------------
    # ANOMALY DETECTION
    # -----------------------
    st.subheader("⚠️ Anomaly Detection")

    try:
        if len(numeric_cols) > 0:
            z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) /
                       df[numeric_cols].std())

            anomalies = df[(z > 3).any(axis=1)]

            st.write(f"Detected {len(anomalies)} anomaly rows")

            if st.checkbox("Show anomalies"):
                st.dataframe(anomalies)

    except Exception:
        st.error("Anomaly detection failed")

    # -----------------------
    # CHARTS
    # -----------------------
    st.subheader("📊 Interactive Dashboard")

    try:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter", "Line", "Bar", "Histogram",
             "Box Plot", "Violin Plot", "Heatmap",
             "Pie Chart", "Pairplot"]
        )

        cols = df.columns.tolist()

        x = st.selectbox("X-axis", cols)

        y = None
        if chart_type not in ["Histogram", "Heatmap", "Pie Chart", "Pairplot"]:
            y = st.selectbox("Y-axis", cols)

        if chart_type == "Pie Chart":
            pie_col = st.selectbox("Category Column", cols)

        if st.button("Generate Chart 🚀"):

            if chart_type == "Scatter":
                fig = px.scatter(filtered_df, x=x, y=y)

            elif chart_type == "Line":
                fig = px.line(filtered_df, x=x, y=y)

            elif chart_type == "Bar":
                fig = px.bar(filtered_df, x=x, y=y)

            elif chart_type == "Histogram":
                fig = px.histogram(filtered_df, x=x)

            elif chart_type == "Box Plot":
                fig = px.box(filtered_df, y=x)

            elif chart_type == "Violin Plot":
                fig = px.violin(filtered_df, y=x, box=True, points="all")

            elif chart_type == "Heatmap":
                fig = px.imshow(filtered_df[numeric_cols].corr(), text_auto=True)

            elif chart_type == "Pie Chart":
                temp = filtered_df[pie_col].value_counts().reset_index()
                temp.columns = [pie_col, "count"]
                fig = px.pie(temp, names=pie_col, values="count")

            elif chart_type == "Pairplot":
                st.info("Generating Pairplot...")
                sns_plot = sns.pairplot(filtered_df[numeric_cols])
                st.pyplot(sns_plot)
                fig = None

            if chart_type != "Pairplot":
                st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.error("Chart generation failed")

    # -----------------------
    # REPORTS (FIXED - NO .BIN ISSUE)
    # -----------------------
    st.subheader("📄 Auto Report Generator")

    try:
        report_type = st.selectbox("Choose Report", ["YData Profiling", "Sweetviz"])

        if st.button("Generate Report"):

            os.makedirs("reports", exist_ok=True)

            # -----------------------
            # YDATA PROFILING
            # -----------------------
            if report_type == "YData Profiling":

                profile = ProfileReport(df, explorative=True)
                path = "reports/ydata.html"
                profile.to_file(path)

                with open(path, "r", encoding="utf-8") as f:
                    html_data = f.read()

                st.success("📄 YData Report Ready")

                st.download_button(
                    "⬇ Download YData Report",
                    html_data,
                    file_name="ydata_report.html",
                    mime="text/html"
                )

            # -----------------------
            # SWEETVIZ
            # -----------------------
            else:

                report = sv.analyze(df)
                path = "reports/sweetviz.html"

                report.show_html(filepath=path, open_browser=False)

                with open(path, "r", encoding="utf-8") as f:
                    html_data = f.read()

                st.success("📄 Sweetviz Report Ready")

                st.download_button(
                    "⬇ Download Sweetviz Report",
                    html_data,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )

    except Exception:
        st.error("Report generation failed")

else:
    st.info("👆 Please upload a CSV file to start")