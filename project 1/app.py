import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Google Play Store Analysis",
    page_icon="📱",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(to right,#f7fbff,#eef4ff);
}
h1{
    color:#1E3A8A;
    text-align:center;
}
h2,h3{
    color:#2563EB;
}
[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #dbeafe;
}
section[data-testid="stSidebar"]{
    background:#1E293B;
}
section[data-testid="stSidebar"] *{
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
<h1>📱 Google Play Store Analysis Dashboard</h1>
<p style="text-align:center;color:gray;font-size:18px;">
Interactive Dashboard using Streamlit
</p>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore_v2.csv")

    # Rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Reviews
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # Price
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("$","",regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Installs
    df["Installs"] = (
        df["Installs"]
        .astype(str)
        .str.replace(",","",regex=False)
        .str.replace("+","",regex=False)
    )
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    # Size
    df["Size"] = (
        df["Size"]
        .astype(str)
        .str.replace("M","",regex=False)
        .str.replace("k","",regex=False)
        .str.replace("Varies with device","",regex=False)
    )
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

    return df

df = load_data()

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.header("📋 Dataset Preview")

st.dataframe(df, use_container_width=True)

# -----------------------------
# DATASET INFO
# -----------------------------
st.header("📊 Dataset Information")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Categories", df["Category"].nunique())
c4.metric("Average Rating", round(df["Rating"].mean(),2))

st.subheader("Column Types")
st.dataframe(
    df.dtypes.astype(str),
    use_container_width=True
)

# -----------------------------
# MISSING VALUES
# -----------------------------
st.header("🧹 Missing Values")

missing = (
    df.isnull()
      .sum()
      .reset_index()
)

missing.columns=["Column","Missing Values"]

st.dataframe(
    missing,
    use_container_width=True
)

# -----------------------------
# CLEANING
# -----------------------------
df = df.dropna(subset=["Rating"])

st.success("✅ Dataset cleaned successfully.")

# -----------------------------
# SUMMARY
# -----------------------------
st.header("📈 Summary Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# -----------------------------
# VISUALIZATION MENU
# -----------------------------
st.header("📊 Visualizations")

option = st.selectbox(
    "Choose Chart",
    [
        "Rating Distribution",
        "Price Distribution",
        "Top Categories",
        "Content Rating",
        "Rating vs Reviews"
    ]
)
# -----------------------------
# CHARTS
# -----------------------------

if option == "Rating Distribution":

    chart = (
        alt.Chart(df)
        .mark_bar(color="#2563EB")
        .encode(
            alt.X("Rating:Q", bin=alt.Bin(maxbins=20), title="Rating"),
            alt.Y("count()", title="Number of Apps"),
            tooltip=["count()"]
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

elif option == "Price Distribution":

    chart = (
        alt.Chart(df)
        .mark_bar(color="#16A34A")
        .encode(
            alt.X("Price:Q", bin=alt.Bin(maxbins=20), title="Price ($)"),
            alt.Y("count()", title="Number of Apps"),
            tooltip=["count()"]
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

elif option == "Top Categories":

    top_categories = (
        df["Category"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_categories.columns = ["Category", "Count"]

    chart = (
        alt.Chart(top_categories)
        .mark_bar(color="#7C3AED")
        .encode(
            x=alt.X("Category:N", sort="-y"),
            y="Count:Q",
            tooltip=["Category", "Count"]
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

elif option == "Content Rating":

    content = (
        df["Content Rating"]
        .value_counts()
        .reset_index()
    )

    content.columns = ["Content Rating", "Count"]

    chart = (
        alt.Chart(content)
        .mark_bar(color="#EA580C")
        .encode(
            x="Content Rating:N",
            y="Count:Q",
            tooltip=["Content Rating", "Count"]
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

elif option == "Rating vs Reviews":

    scatter = df.dropna(subset=["Reviews", "Rating"])

    chart = (
        alt.Chart(scatter)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("Reviews:Q", title="Reviews"),
            y=alt.Y("Rating:Q", title="Rating"),
            color=alt.Color("Rating:Q"),
            tooltip=[
                "App",
                "Category",
                "Rating",
                "Reviews"
            ]
        )
        .properties(height=500)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# DATA INSIGHTS
# -----------------------------

st.header("📌 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.info(f"⭐ Average Rating: **{df['Rating'].mean():.2f}**")
    st.info(f"📂 Total Categories: **{df['Category'].nunique()}**")

with col2:
    st.info(f"📱 Total Apps: **{len(df):,}**")
    st.info(f"👨‍💻 Total Reviews: **{int(df['Reviews'].fillna(0).sum()):,}**")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
padding:20px;
font-size:16px;
color:gray;'>

❤️ Developed using <b>Streamlit</b><br>
Google Play Store Analysis Dashboard

</div>
""",
unsafe_allow_html=True
)
